import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
import cv2
import pandas as pd
import pickle
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
# barf
import camera

class PhototourismDataset(Dataset):
    def __init__(self, root_dir, feat_dir=None, pca_info_dir=None, camera_noise=0.0, N_vocab=1500, split='train', img_downscale=1, fewshot=-1, img_idx=[0], use_cache=False):
        """
        img_downscale: how much scale to downsample the training images.
                       The original image sizes are around 500~100, so value of 1 or 2
                       are recommended.
                       ATTENTION! Value of 1 will consume large CPU memory,
                       about 40G for brandenburg gate.
        val_num: number of val images (used for multigpu, validate same image for all gpus)
        use_cache: during data preparation, use precomputed rays (useful to accelerate
                   data loading, especially for multigpu!)
        """
        self.root_dir = root_dir
        self.feat_dir = feat_dir
        self.pca_info_dir = pca_info_dir
        self.camera_noise = camera_noise
        self.N_vocab = N_vocab
        self.split = split
        assert img_downscale >= 1, 'image can only be downsampled, please set img_downscale>=1!'
        self.img_downscale = img_downscale
        if split == 'val': # image downscale=1 will cause OOM in val mode
            self.img_downscale = max(2, self.img_downscale)
        self.fewshot = fewshot
        self.img_idx = img_idx
        self.use_cache = use_cache
        self.define_transforms()

        self.read_meta()
        self.white_back = False

    def read_meta(self):
        # read all files in the tsv first (split to train and test later)
        tsv = glob.glob(os.path.join(self.root_dir, '*.tsv'))[0]
        self.scene_name = os.path.basename(tsv)[:-4]
        self.files = pd.read_csv(tsv, sep='\t')
        self.files = self.files[~self.files['id'].isnull()] # remove data without id
        self.files.reset_index(inplace=True, drop=True)

        # Step 1. load image paths
        # Attention! The 'id' column in the tsv is BROKEN, don't use it!!!!
        # Instead, read the id from images.bin using image file name!
        if self.use_cache:
            with open(os.path.join(self.root_dir, f'cache/img_ids.pkl'), 'rb') as f:
                self.img_ids = pickle.load(f)
            with open(os.path.join(self.root_dir, f'cache/image_paths.pkl'), 'rb') as f:
                self.image_paths = pickle.load(f)
        else:
            imdata = read_images_binary(os.path.join(self.root_dir, 'dense/sparse/images.bin'))
            img_path_to_id = {}
            for v in imdata.values():
                img_path_to_id[v.name] = v.id
            self.img_ids = []
            self.image_paths = {} # {id: filename}
            for filename in list(self.files['filename']):
                id_ = img_path_to_id[filename]
                self.image_paths[id_] = filename
                self.img_ids += [id_]

        # Step 2: read and rescale camera intrinsics
        if self.use_cache:
            with open(os.path.join(self.root_dir, f'cache/Ks{self.img_downscale}.pkl'), 'rb') as f:
                self.Ks = pickle.load(f)
        else:
            self.Ks = {} # {id: K}
            camdata = read_cameras_binary(os.path.join(self.root_dir, 'dense/sparse/cameras.bin'))
            for id_ in self.img_ids:
                K = np.zeros((3, 3), dtype=np.float32)
                cam = camdata[id_]
                img_w, img_h = int(cam.params[2]*2), int(cam.params[3]*2)
                img_w_, img_h_ = img_w//self.img_downscale, img_h//self.img_downscale
                K[0, 0] = cam.params[0]*img_w_/img_w # fx
                K[1, 1] = cam.params[1]*img_h_/img_h # fy
                K[0, 2] = cam.params[2]*img_w_/img_w # cx
                K[1, 2] = cam.params[3]*img_h_/img_h # cy
                K[2, 2] = 1
                self.Ks[id_] = K

        # Step 3: read c2w poses (of the images in tsv file only) and correct the order
        if self.use_cache:
            self.poses = np.load(os.path.join(self.root_dir, 'cache/poses.npy'))
        else:
            w2c_mats = []
            bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
            for id_ in self.img_ids:
                im = imdata[id_]
                R = im.qvec2rotmat()
                t = im.tvec.reshape(3, 1)
                w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
            w2c_mats = np.stack(w2c_mats, 0) # (N_images, 4, 4)
            self.poses = np.linalg.inv(w2c_mats)[:, :3] # (N_images, 3, 4)
            # Original poses has rotation in form "right down front", change to "right up back"
            self.poses[..., 1:3] *= -1

        # Step 4: correct scale
        if self.use_cache:
            self.xyz_world = np.load(os.path.join(self.root_dir, 'cache/xyz_world.npy'))
            with open(os.path.join(self.root_dir, f'cache/nears.pkl'), 'rb') as f:
                self.nears = pickle.load(f)
            with open(os.path.join(self.root_dir, f'cache/fars.pkl'), 'rb') as f:
                self.fars = pickle.load(f)
        else:
            pts3d = read_points3d_binary(os.path.join(self.root_dir, 'dense/sparse/points3D.bin'))
            self.xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
            xyz_world_h = np.concatenate([self.xyz_world, np.ones((len(self.xyz_world), 1))], -1)
            # Compute near and far bounds for each image individually
            self.nears, self.fars = {}, {} # {id_: distance}
            for i, id_ in enumerate(self.img_ids):
                xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[:, :3] # xyz in the ith cam coordinate
                xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2]>0] # filter out points that lie behind the cam
                self.nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1)
                self.fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)

            max_far = np.fromiter(self.fars.values(), np.float32).max()
            scale_factor = max_far/5 # so that the max far is scaled to 5
            self.poses[..., 3] /= scale_factor
            for k in self.nears:
                self.nears[k] /= scale_factor
            for k in self.fars:
                self.fars[k] /= scale_factor
            self.xyz_world /= scale_factor
        self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)}
            
        # Step 5. split the img_ids (the number of images is verfied to match that in the paper)
        self.img_ids_train = [id_ for i, id_ in enumerate(self.img_ids) 
                                    if self.files.loc[i, 'split']=='train']
        self.img_ids_test = [id_ for i, id_ in enumerate(self.img_ids)
                                    if self.files.loc[i, 'split']=='test']
        self.N_images_train = len(self.img_ids_train)
        self.N_images_test = len(self.img_ids_test)

        # barf
        poses = []
        try:
            self.id2idx = torch.zeros(self.N_vocab, dtype=torch.long)
            for idx, id_ in enumerate(self.img_ids_train):
            # for idx, (k, v) in enumerate(self.poses_dict.items()):
                self.id2idx[id_] = idx
                poses += [self.poses_dict[id_]]
            poses = np.stack(poses, 0)
        except:
            max_id = max(self.img_ids_train)
            raise Exception(f"N_vocab must be larger than dataset_num({max_id})")
        poses = torch.FloatTensor(poses)
        
        if self.camera_noise is not None:
            self.GT_poses_dict = self.poses_dict
            noise_file_name = f'noises/{len(poses)}_{str(self.camera_noise)}.pt'
            if os.path.isfile(noise_file_name):
                self.pose_noises = torch.load(noise_file_name)
                poses = camera.pose.compose([self.pose_noises, poses])
                self.poses_dict = {id_: poses[i] for i, id_ in enumerate(self.img_ids_train)}
                print("load noise file: ", noise_file_name)
            else:
                # pre-generate pose perturbation
                if self.camera_noise == -1:
                    # intialize poses at one view point.
                    self.poses_dict = {id_: torch.eye(3,4) for i, id_ in enumerate(self.img_ids_train)}
                else:
                    se3_noise = torch.randn(len(poses),6)*self.camera_noise
                    self.pose_noises = camera.lie.se3_to_SE3(se3_noise)
                    torch.save(self.pose_noises, noise_file_name)
                    poses = camera.pose.compose([self.pose_noises, poses])
                    self.poses_dict = {id_: poses[i] for i, id_ in enumerate(self.img_ids_train)}
                

        if self.split == 'train': # create buffer of all rays and rgb data
            if self.use_cache:
                all_ray_infos = np.load(os.path.join(self.root_dir,
                                                f'cache/ray_infos{self.img_downscale}.npy'))
                self.all_ray_infos = torch.from_numpy(all_ray_infos)
                all_rgbs = np.load(os.path.join(self.root_dir,
                                                f'cache/rgbs{self.img_downscale}.npy'))
                self.all_rgbs = torch.from_numpy(all_rgbs)
                all_directions = np.load(os.path.join(self.root_dir,
                                                f'cache/directions{self.img_downscale}.npy'))
                self.all_directions = torch.from_numpy(all_directions)
                
                if self.fewshot != -1:
                    few_ray_infos = []
                    few_rgbs = []
                    few_directions = []
                    all_imgs_wh = np.load(os.path.join(self.root_dir,
                                                f'cache/all_imgs_wh{self.img_downscale}.npy'))
                    self.all_imgs_wh = torch.from_numpy(all_imgs_wh)
                    txt_path = os.path.join(self.root_dir, f"few_{self.fewshot}.txt")
                    with open(txt_path, 'r') as f:
                        few_list = f.read().splitlines()
                    idcs = [i for i,id_ in enumerate(self.img_ids_train) if self.image_paths[id_] in few_list]
                    self.img_ids_train = few_list
                    for idc in idcs:
                        img_w, img_h = self.all_imgs_wh[idc].long()
                        s_id = (self.all_imgs_wh[:idc, 0]*self.all_imgs_wh[:idc, 1]).sum().long().item()
                        few_ray_infos += [self.all_ray_infos[s_id:s_id+img_w.item()*img_h.item()]]
                        few_rgbs += [self.all_rgbs[s_id:s_id+img_w.item()*img_h.item()]]
                        few_directions += [self.all_directions[s_id:s_id+img_w.item()*img_h.item()]]
                    self.all_ray_infos = torch.cat(few_ray_infos, 0)
                    self.all_rgbs = torch.cat(few_rgbs, 0)
                    self.all_directions = torch.cat(few_directions, 0)
                
                
                if self.feat_dir is not None:
                    idcs = [i for i in range(self.N_images_train)]           
                    all_imgs_wh = np.load(os.path.join(self.root_dir,f'cache/all_imgs_wh{self.img_downscale}.npy'))
                    self.feat_names = [self.image_paths[self.img_ids_train[i]].replace('.jpg','.npy') for i in idcs]
                    self.seg_feats = []
                    self.feat_idx = []
                    c_i = 0
                    for f_n, (W,H) in zip(self.feat_names, all_imgs_wh):
                        W, H = int(W), int(H)
                        feat_map = np.load(os.path.join(self.feat_dir, f_n))                  # [C,H',W']
                        C, H_, W_ = feat_map.shape
                        idx_map = np.arange(H_*W_).reshape(H_,W_)/(H_*W_)
                        idx_map = (cv2.resize(idx_map, (W,H), interpolation=cv2.INTER_NEAREST)*(H_*W_)).astype(np.int64)
                        idx_map = torch.from_numpy(idx_map).view(-1)+c_i
                        c_i = c_i+(H_*W_)
                        feat_map = torch.from_numpy(feat_map.transpose(1,2,0)).view(-1,C)
                        feat_map = feat_map/torch.norm(feat_map, dim=-1, keepdim=True)
                        self.seg_feats.append(feat_map)
                        self.feat_idx.append(idx_map)
                    self.seg_feats = torch.cat(self.seg_feats, 0)
                    self.feat_idx = torch.cat(self.feat_idx, 0)


            else:
                self.all_ray_infos = []
                self.all_rgbs = []
                self.all_directions = []
                for id_ in self.img_ids_train:
                    img = Image.open(os.path.join(self.root_dir, 'dense/images',
                                                  self.image_paths[id_])).convert('RGB')
                    img_w, img_h = img.size
                    if self.img_downscale > 1:
                        img_w = img_w//self.img_downscale
                        img_h = img_h//self.img_downscale
                        img = img.resize((img_w, img_h), Image.LANCZOS)
                    img = self.transform(img) # (3, h, w)
                    img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                    self.all_rgbs += [img]
                    
                    directions = get_ray_directions(img_h, img_w, self.Ks[id_]).view(-1,3)
                    self.all_directions += [directions]
                    rays_t = id_ * torch.ones(len(directions), 1)

                    self.all_ray_infos += [torch.cat([
                                                self.nears[id_]*torch.ones_like(directions[:, :1]),
                                                self.fars[id_]*torch.ones_like(directions[:, :1]),
                                                rays_t],
                                                1)] # (h*w, 8)
                
                                    
                self.all_ray_infos = torch.cat(self.all_ray_infos, 0) # ((N_images-1)*h*w, 8)
                self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
                self.all_directions = torch.cat(self.all_directions, 0)
                np.save(os.path.join(self.root_dir,
                            f'cache/ray_infos{self.img_downscale}.npy'), self.all_ray_infos)
                np.save(os.path.join(self.root_dir,
                                            f'cache/directions{self.img_downscale}.npy'), self.all_directions)
                raise
        
        elif self.split in ['val', 'test_train']: # use the first image as val image (also in train)     
            idcs = [i for i in range(self.N_images_train)]        
            self.feat_names = [self.image_paths[self.img_ids_train[i]].replace('.jpg','.npy') for i in idcs]
            if self.fewshot != -1:
                txt_path = os.path.join(self.root_dir, f"few_{self.fewshot}.txt")
                with open(txt_path, 'r') as f:
                    few_list = f.read().splitlines()
                idcs = [i for i,id_ in enumerate(self.img_ids_train) if self.image_paths[id_] in few_list]
                self.img_ids_train = [id_ for i,id_ in enumerate(self.img_ids_train) if self.image_paths[id_] in few_list]
                self.N_images_train = self.fewshot

        else: # for testing, create a parametric rendering path
            # test poses and appearance index are defined in eval.py
            self.poses_test = []
            for id in self.img_ids_test:
                self.poses_test += [self.poses_dict[id]]


    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_ray_infos)
        if self.split == 'test_train':
            return self.N_images_train
        if self.split == 'val':
            return len(self.img_idx)
        if self.split == 'test':
            return self.N_images_test
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            ts = self.all_ray_infos[idx, 2].long()
            ts_idx = self.id2idx[ts]
            sample = {'ray_infos': self.all_ray_infos[idx, :2],
                      'directions': self.all_directions[idx], 
                      'ts': ts,
                      'ts_idx': ts_idx, 
                      'c2w': torch.FloatTensor(self.poses_dict[ts.item()]),
                      'rgbs': self.all_rgbs[idx]}
            if self.feat_dir is not None:
                sample['feats'] = self.seg_feats[self.feat_idx[idx]]
            
        elif self.split == 'val':
            sample = {}
            idx = self.img_idx[idx]
            id_ = self.img_ids_train[idx]

            sample['c2w'] = c2w = torch.FloatTensor(self.poses_dict[id_])
            img = Image.open(os.path.join(self.root_dir, 'dense/images',
                                          self.image_paths[id_])).convert('RGB')
            img_w, img_h = img.size
            if self.img_downscale > 1:
                img_w = img_w//self.img_downscale
                img_h = img_h//self.img_downscale
                img = img.resize((img_w, img_h), Image.LANCZOS)
            img = self.transform(img) # (3, h, w)
            img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
            sample['rgbs'] = img

            directions = get_ray_directions(img_h, img_w, self.Ks[id_]).view(-1, 3)
            # rays_o, rays_d = get_rays(directions, c2w)
            ray_infos = torch.cat([
                              self.nears[id_]*torch.ones_like(directions[:, :1]),
                              self.fars[id_]*torch.ones_like(directions[:, :1])],
                              1) # (h*w, 8)
            sample['ray_infos'] = ray_infos
            sample['directions'] = directions
            sample['ts'] = id_ * torch.ones(len(ray_infos), dtype=torch.long)
            sample['ts_idx'] = self.id2idx[sample['ts']]
            sample['img_wh'] = torch.LongTensor([img_w, img_h])
            
            if self.feat_dir is not None:
                feat_map = np.load(os.path.join(self.feat_dir,self.feat_names[idx]))                  # [C,H',W']
                feat_map = feat_map = feat_map/np.linalg.norm(feat_map, axis=0, keepdims=True)
                feat_map = torch.from_numpy(cv2.resize(feat_map.transpose(1,2,0),(img_w,img_h), interpolation=cv2.INTER_NEAREST))  # [H,
                m = np.load(self.pca_info_dir+self.feat_names[idx].replace('.npy','_mean.npy'))
                c = np.load(self.pca_info_dir+self.feat_names[idx].replace('.npy','_components.npy'))
                sample['feats'] = feat_map
                sample['m'] = torch.from_numpy(m)
                sample['c'] = torch.from_numpy(c)
                
        elif self.split == 'test_train':
            sample = {}
            id_ = self.img_ids_train[idx]

            sample['c2w'] = c2w = torch.FloatTensor(self.poses_dict[id_])
            img = Image.open(os.path.join(self.root_dir, 'dense/images',
                                          self.image_paths[id_])).convert('RGB')
            img_w, img_h = img.size
            if self.img_downscale > 1:
                img_w = img_w//self.img_downscale
                img_h = img_h//self.img_downscale
                img = img.resize((img_w, img_h), Image.LANCZOS)
            img = self.transform(img) # (3, h, w)
            img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
            sample['rgbs'] = img

            directions = get_ray_directions(img_h, img_w, self.Ks[id_]).view(-1, 3)
            # rays_o, rays_d = get_rays(directions, c2w)
            ray_infos = torch.cat([
                              self.nears[id_]*torch.ones_like(directions[:, :1]),
                              self.fars[id_]*torch.ones_like(directions[:, :1])],
                              1) # (h*w, 8)
            sample['ray_infos'] = ray_infos
            sample['directions'] = directions
            sample['ts'] = id_ * torch.ones(len(ray_infos), dtype=torch.long)
            sample['ts_idx'] = self.id2idx[sample['ts']]
            sample['img_wh'] = torch.LongTensor([img_w, img_h])
            
            if self.feat_dir is not None:
                feat_map = np.load(os.path.join(self.feat_dir,self.feat_names[idx]))                  # [C,H',W']
                feat_map = feat_map = feat_map/np.linalg.norm(feat_map, axis=0, keepdims=True)
                feat_map = torch.from_numpy(cv2.resize(feat_map.transpose(1,2,0),(img_w,img_h), interpolation=cv2.INTER_NEAREST))  # [H,
                m = np.load(self.pca_info_dir+self.feat_names[idx].replace('.npy','_mean.npy'))
                c = np.load(self.pca_info_dir+self.feat_names[idx].replace('.npy','_components.npy'))
                sample['feats'] = feat_map
                sample['m'] = torch.from_numpy(m)
                sample['c'] = torch.from_numpy(c)

        elif self.split == 'video':
            sample = {}
            sample['c2w'] = c2w = torch.FloatTensor(self.poses_test[idx])
            directions = get_ray_directions(self.test_img_h, self.test_img_w, self.test_K)
            rays_o, rays_d = get_rays(directions, c2w)
            near, far = 0, 5
            rays = torch.cat([rays_o, rays_d,
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1)
            sample['rays'] = rays
            sample['ts'] = self.test_appearance_idx * torch.ones(len(rays), dtype=torch.long)
            sample['img_wh'] = torch.LongTensor([self.test_img_w, self.test_img_h])
            
            
        else: # test
            
            sample = {}
            sample['c2w'] = c2w = torch.FloatTensor(self.poses_test[idx])
            id_ = self.img_ids_test[idx]
            img = Image.open(os.path.join(self.root_dir, 'dense/images',
                                          self.image_paths[id_])).convert('RGB')
            img_w, img_h = img.size
            if self.img_downscale > 1:
                img_w = img_w//self.img_downscale
                img_h = img_h//self.img_downscale
                img = img.resize((img_w, img_h), Image.LANCZOS)
            img = self.transform(img) # (3, h, w)
            img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
            sample['rgbs'] = img

            directions = get_ray_directions(img_h, img_w, self.Ks[id_])
            # directions = get_ray_directions(self.test_img_h, self.test_img_w, self.test_K)
            rays_o, rays_d = get_rays(directions, c2w)
            rays = torch.cat([rays_o, rays_d,
                              self.nears[id_]*torch.ones_like(rays_o[:, :1]),
                              self.fars[id_]*torch.ones_like(rays_o[:, :1])],
                              1) # (h*w, 8)
            # near, far = 0, 5
            # rays = torch.cat([rays_o, rays_d,
            #                   near*torch.ones_like(rays_o[:, :1]),
            #                   far*torch.ones_like(rays_o[:, :1])],
            #                   1)
            sample['rays'] = rays
            sample['ts'] = self.test_appearance_idx * torch.ones(len(rays), dtype=torch.long)
            sample['img_wh'] = torch.LongTensor([img_w, img_h])

        return sample



class PhototourismOptimizeDataset(Dataset):
    def __init__(self, root_dir, split='train', img_downscale=1, use_cache=False, batch_size=1024, scale_anneal=-1, min_scale=0.25, encode_random=True, data_idx=0):
        """
        img_downscale: how much scale to downsample the training images.
                       The original image sizes are around 500~100, so value of 1 or 2
                       are recommended.
                       ATTENTION! Value of 1 will consume large CPU memory,
                       about 40G for brandenburg gate.
        use_cache: during data preparation, use precomputed rays (useful to accelerate
                   data loading, especially for multigpu!)
        """
        self.data_idx = data_idx
        self.root_dir = root_dir
        self.split = split
        assert img_downscale >= 1, 'image can only be downsampled, please set img_downscale>=1!'
        self.img_downscale = img_downscale

        if split == 'val': # image downscale=1 will cause OOM in val mode
            self.img_downscale = max(2, self.img_downscale)
        self.use_cache = use_cache
        self.define_transforms()

        self.read_meta()
        self.white_back = False

        # no effect if scale_anneal<0, else the minimum scale decreases exponentially until converge to min_scale
        self.scale_anneal = scale_anneal
        self.min_scale = min_scale

    def read_meta(self):
        # read all files in the tsv first (split to train and test later)
        tsv = glob.glob(os.path.join(self.root_dir, '*.tsv'))[0]
        self.scene_name = os.path.basename(tsv)[:-4]
        self.files = pd.read_csv(tsv, sep='\t')
        # self.files = pd.read_csv(tsv)
        self.files = self.files[~self.files['id'].isnull()] # remove data without id
        self.files.reset_index(inplace=True, drop=True)
        if self.use_cache:
            with open(os.path.join(self.root_dir, f'cache/img_ids.pkl'), 'rb') as f:
                self.img_ids = pickle.load(f)
            with open(os.path.join(self.root_dir, f'cache/image_paths.pkl'), 'rb') as f:
                self.image_paths = pickle.load(f)
            self.img_ids_test = [id_ for i, id_ in enumerate(self.img_ids) 
                                    if self.files.loc[i, 'split']=='test']
        else:
            imdata = read_images_binary(os.path.join(self.root_dir, 'dense/sparse/images.bin'))
            img_path_to_id = {}
            for v in imdata.values():
                img_path_to_id[v.name] = v.id
            self.img_ids = []
            self.image_paths = {} # {id: filename}
            for filename in list(self.files['filename']):
                id_ = img_path_to_id[filename]
                self.image_paths[id_] = filename
                self.img_ids += [id_]
            self.img_ids_test = [id_ for i, id_ in enumerate(self.img_ids) 
                                    if self.files.loc[i, 'split']=='test']
            self.Ks = {} # {id: K}
            camdata = read_cameras_binary(os.path.join(self.root_dir, 'dense/sparse/cameras.bin'))
            for id_ in self.img_ids_test:
                K = np.zeros((3, 3), dtype=np.float32)
                cam = camdata[id_]
                img_w, img_h = int(cam.params[2]*2), int(cam.params[3]*2)
                img_w_, img_h_ = img_w//self.img_downscale, img_h//self.img_downscale
                K[0, 0] = cam.params[0]*img_w_/img_w # fx
                K[1, 1] = cam.params[1]*img_h_/img_h # fy
                K[0, 2] = cam.params[2]*img_w_/img_w # cx
                K[1, 2] = cam.params[3]*img_h_/img_h # cy
                K[2, 2] = 1
                self.Ks[id_] = K

        
        self.N_images_test = len(self.img_ids_test)
        self.img_names_test = [self.image_paths[id_].replace('.jpg','') for id_ in self.img_ids_test]

        # if self.use_cache:
        #     import pdb;pdb.set_trace()
        #     self.poses = np.load(os.path.join(self.root_dir, 'cache/poses.npy'))
        #     self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)}
        #     with open(os.path.join(self.root_dir, f'cache/Ks{self.img_downscale}.pkl'), 'rb') as f:
        #         self.Ks = pickle.load(f)
        #     self.xyz_world = np.load(os.path.join(self.root_dir, 'cache/xyz_world.npy'))
        #     with open(os.path.join(self.root_dir, f'cache/nears.pkl'), 'rb') as f:
        #         self.nears = pickle.load(f)
        #     with open(os.path.join(self.root_dir, f'cache/fars.pkl'), 'rb') as f:
        #         self.fars = pickle.load(f)
        
        w2c_mats = []
        bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
        for id_ in self.img_ids_test:
            im = imdata[id_]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0) # (N_images, 4, 4)
        self.poses = np.linalg.inv(w2c_mats)[:, :3] # (N_images, 3, 4)
        # Original poses has rotation in form "right down front", change to "right up back"
        self.poses[..., 1:3] *= -1
        
        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'dense/sparse/points3D.bin'))
        self.xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
        xyz_world_h = np.concatenate([self.xyz_world, np.ones((len(self.xyz_world), 1))], -1)
        # Compute near and far bounds for each image individually
        self.nears, self.fars = {}, {} # {id_: distance}
        for i, id_ in enumerate(self.img_ids_test):
            xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[:, :3] # xyz in the ith cam coordinate
            xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2]>0] # filter out points that lie behind the cam
            self.nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1)
            self.fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)

        max_far = np.fromiter(self.fars.values(), np.float32).max()
        scale_factor = max_far/5 # so that the max far is scaled to 5
        self.poses[..., 3] /= scale_factor
        for k in self.nears:
            self.nears[k] /= scale_factor
        for k in self.fars:
            self.fars[k] /= scale_factor
        self.xyz_world /= scale_factor
        self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids_test)}
        
        

        

        if self.split == 'train':
            self.all_left_rays = []
            self.all_left_rgbs = []
            self.all_left_hw = []
            for id_ in self.img_ids_test[self.data_idx:self.data_idx+1]:
                c2w = torch.FloatTensor(self.poses_dict[id_])

                img = Image.open(os.path.join(self.root_dir, 'dense/images',
                                                self.image_paths[id_])).convert('RGB')
                img_w, img_h = img.size
                if self.img_downscale > 1:
                    img_w = img_w//self.img_downscale
                    img_h = img_h//self.img_downscale
                    img = img.resize((img_w, img_h), Image.LANCZOS)
                self.img_w, self.img_h = img_w//2, img_h
                img = self.transform(img) # (3, h, w)
                left_img = img[:,:,:-img_w//2]
                                
                self.all_left_hw += [(img_h, left_img.shape[2])]
            
                left_img = left_img.reshape(3, -1).permute(1, 0) # (h//2*w, 3) RGB
                self.all_left_rgbs += [left_img]
                
                
                directions = get_ray_directions(img_h, img_w, self.Ks[id_])

                left_directions = directions[:,:-img_w//2,:]
                left_rays_d = left_directions @ c2w[:, :3].T # (H, W, 3)
                # The origin of all rays is the camera origin in world coordinate
                left_rays_o = c2w[:, 3].expand(left_rays_d.shape) # (H, W, 3)
                left_rays_d = left_rays_d.reshape(-1, 3)
                left_rays_o = left_rays_o.reshape(-1, 3)


                left_rays_t = id_ * torch.ones(len(left_rays_o), 1)
                self.all_left_rays += [torch.cat([left_rays_o, left_rays_d,
                                            self.nears[id_]*torch.ones_like(left_rays_o[:, :1]),
                                            self.fars[id_]*torch.ones_like(left_rays_o[:, :1]),
                                            left_rays_t],
                                            1)] # (h*w, 9)
            self.all_left_rays = torch.cat(self.all_left_rays, 0) # ((N_images-1)*h*w, 9)
            self.all_left_rgbs = torch.cat(self.all_left_rgbs, 0) # ((N_images-1)*h*w, 3)

        elif self.split == 'val':
            self.all_right_rays = []
            self.all_right_rgbs = []
            self.all_right_hw = []
            scale_down = 1
            for id_ in self.img_ids_test[self.data_idx:self.data_idx+1]:
                c2w = torch.FloatTensor(self.poses_dict[id_])

                img = Image.open(os.path.join(self.root_dir, 'dense/images',
                                                self.image_paths[id_])).convert('RGB')
                
                img_w, img_h = img.size
                if self.img_downscale > 1:
                    img_w = img_w//self.img_downscale
                    img_h = img_h//self.img_downscale
                    img = img.resize((img_w, img_h), Image.LANCZOS)
                    
                
                self.img_w, self.img_h = img_w - img_w//2, img_h
                img = self.transform(img) # (3, h, w)
                right_img = img[:,:,-img_w//2:]
                self.all_right_hw += [(img_h, right_img.shape[2])]
            
                right_img = right_img[:, ::scale_down, ::scale_down].reshape(3, -1).permute(1, 0) # (h//2*w, 3) RGB
                self.all_right_rgbs += [right_img]               
                
                directions = get_ray_directions(img_h, img_w, self.Ks[id_])

                right_directions = directions[:,-img_w//2:,:]
                right_rays_d = right_directions @ c2w[:, :3].T # (H, W, 3)
                right_rays_d = right_rays_d[::scale_down, ::scale_down]
                # The origin of all rays is the camera origin in world coordinate
                right_rays_o = c2w[:, 3].expand(right_rays_d.shape) # (H, W, 3)
                right_rays_d = right_rays_d.reshape(-1, 3)
                right_rays_o = right_rays_o.reshape(-1, 3)


                right_rays_t = id_ * torch.ones(len(right_rays_o), 1)
                rays = torch.cat([right_rays_o, right_rays_d,
                                            self.nears[id_]*torch.ones_like(right_rays_o[:, :1]),
                                            self.fars[id_]*torch.ones_like(right_rays_o[:, :1]),
                                            right_rays_t],
                                            1) # (h*w, 9)
                self.all_right_rays += [rays] # (h*w//10, 9)
            self.all_right_rays = torch.cat(self.all_right_rays, 0) # ((N_images-1)*h*w, 9)
            self.all_right_rgbs = torch.cat(self.all_right_rgbs, 0) # ((N_images-1)*h*w, 3)

    def define_transforms(self):
        self.transform = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        if self.split == 'train':
            self.iterations = len(self.all_left_rays)
            return self.iterations
        if self.split == 'val':
            self.iterations = len(self.all_right_rays)
            return self.iterations

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_left_rays[idx, :8],
                      'ts': self.all_left_rays[idx, 8].long(),
                      'rgbs': self.all_left_rgbs[idx]}
        elif self.split == 'val':
            sample = {'rays': self.all_right_rays[idx, :8],
                      'ts': self.all_right_rays[idx, 8].long(),
                      'rgbs': self.all_right_rgbs[idx]}
        return sample
