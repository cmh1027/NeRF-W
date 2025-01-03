import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image, ImageDraw
from torchvision import transforms as T

from .ray_utils import *

# barf
import camera

def add_perturbation(img, perturbation, seed):
    if 'color' in perturbation:
        np.random.seed(seed)
        img_np = np.array(img)/255.0
        s = np.random.uniform(0.8, 1.2, size=3)
        b = np.random.uniform(-0.2, 0.2, size=3)
        img_np[..., :3] = np.clip(s*img_np[..., :3]+b, 0, 1)
        img = Image.fromarray((255*img_np).astype(np.uint8))
    if 'occ' in perturbation:
        draw = ImageDraw.Draw(img)
        np.random.seed(seed)
        left = np.random.randint(200, 400)
        top = np.random.randint(200, 400)
        for i in range(10):
            np.random.seed(10*seed+i)
            random_color = tuple(np.random.choice(range(256), 3))
            draw.rectangle(((left+20*i, top), (left+20*(i+1), top+200)),
                            fill=random_color)
    return img


class BlenderDataset(Dataset):
    def __init__(self, root_dir, feat_dir=None, pca_info_dir=None, split='train', img_wh=(800, 800), perturbation=[], camera_noise=0.0, img_idx=[0]):
        self.root_dir = root_dir
        self.feat_dir = feat_dir
        self.pca_info_dir = pca_info_dir
        # self.split = split
        self.split = 'test_train' if split == 'val' else split
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'
        self.img_wh = img_wh
        self.define_transforms()
        assert set(perturbation).issubset({"color", "occ"}), 'Only "color" and "occ" perturbations are supported!'
        self.perturbation = perturbation
        if self.split == 'train' and len(self.perturbation) > 0 :
            print(f'add {self.perturbation} perturbation!')
        self.camera_noise = camera_noise
        self.read_meta()
        self.white_back = True
        self.img_idx = img_idx

    def read_meta(self):
        with open(os.path.join(self.root_dir, f"transforms_{self.split.split('_')[-1]}.json"), 'r') as f:
            self.meta = json.load(f)
        w, h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=800

        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh
        self.K = np.eye(3)
        self.K[0, 0] = self.K[1, 1] = self.focal
        self.K[0, 2] = w/2
        self.K[1, 2] = h/2

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, self.K) # (h, w, 3)

        _poses = [f['transform_matrix'] for f in self.meta['frames']]
        self.poses = torch.FloatTensor(_poses)[:, :3, :4]
        
        if self.camera_noise is not None:
            len_ = len(self.meta['frames'])
            self.gt_poses = self.poses
            if self.camera_noise < 0:
                self.poses = torch.eye(3,4)[None, ...].repeat(len_, 1, 1)
                if self.camera_noise != -1:
                    se3_noise = torch.randn(len_,6)*abs(self.camera_noise)
                    self.pose_noises = camera.lie.se3_to_SE3(se3_noise)
                    self.poses = camera.pose.compose([self.pose_noises, self.poses])

            else:
                noise_file_name = f'noises/blender_{len_}_{str(self.camera_noise)}.pt'
                if os.path.isfile(noise_file_name):
                    self.pose_noises = torch.load(noise_file_name)
                    print("load noise file: ", noise_file_name)
                else:
                    se3_noise = torch.randn(len_,6)*self.camera_noise
                    self.pose_noises = camera.lie.se3_to_SE3(se3_noise)
                    torch.save(self.pose_noises, noise_file_name)
                self.poses = camera.pose.compose([self.pose_noises, self.gt_poses])
            
            
        if self.split == 'train': # create buffer of all rays and rgb data
            self.all_ray_infos = []
            self.all_rgbs = []
            self.all_directions = []
            for t, frame in enumerate(self.meta['frames']):
                # pose = np.array(frame['transform_matrix'])[:3, :4]
                # c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                img = Image.open(image_path)
                if t != 0: # perturb everything except the first image.
                           # cf. Section D in the supplementary material
                    img = add_perturbation(img, self.perturbation, t)

                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, h, w)
                img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                self.all_rgbs += [img]
                
                # rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                rays_t = t * torch.ones(len(img), 1)
                self.all_directions += [self.directions.view(-1,3)]
                self.all_ray_infos += [torch.cat([self.near*torch.ones_like(img[:, :1]),
                                             self.far*torch.ones_like(img[:, :1]),
                                             rays_t],
                                             1)] # (h*w, 3)
            # self.all_rays = []
            # self.all_rgbs = []
            # for t, frame in enumerate(self.meta['frames']):
            #     # pose = np.array(frame['transform_matrix'])[:3, :4]
            #     # c2w = torch.FloatTensor(pose)
            #     c2w = self.pose[t]

            #     image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            #     img = Image.open(image_path)
            #     if t != 0: # perturb everything except the first image.
            #                # cf. Section D in the supplementary material
            #         img = add_perturbation(img, self.perturbation, t)

            #     img = img.resize(self.img_wh, Image.LANCZOS)
            #     img = self.transform(img) # (4, h, w)
            #     img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
            #     img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
            #     self.all_rgbs += [img]
                
            #     rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
            #     rays_t = t * torch.ones(len(rays_o), 1)

            #     self.all_rays += [torch.cat([rays_o, rays_d,
            #                                  self.near*torch.ones_like(rays_o[:, :1]),
            #                                  self.far*torch.ones_like(rays_o[:, :1]),
            #                                  rays_t],
            #                                  1)] # (h*w, 8)

            self.all_ray_infos = torch.cat(self.all_ray_infos, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_directions = torch.cat(self.all_directions, 0) # (len(self.meta['frames])*h*w, 3)
            
        self.N_images_train = len(self.poses)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_ray_infos)
        elif self.split == 'val' or self.split == 'test_train':
            return len(self.img_idx) # only validate 8 images (to support <=8 gpus)
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            ts = self.all_ray_infos[idx, 2].long()
            sample = {'ray_infos': self.all_ray_infos[idx, :2],
                      'directions': self.all_directions[idx], 
                      'ts': ts,
                      'c2w': self.poses[ts],
                      'rgbs': self.all_rgbs[idx]}

        else: # create data for each image separately
            frame = self.meta['frames'][idx]
            c2w = self.poses[idx]
            directions = self.directions.view(-1,3)
            t = 0 # transient embedding index, 0 for val and test (no perturbation)

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            if self.split == 'test_train' and idx != 0:
                t = idx
                img = add_perturbation(img, self.perturbation, idx)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, H, W)
            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

            ray_infos = torch.cat([
                              self.near*torch.ones_like(directions[:, :1]),
                              self.far*torch.ones_like(directions[:, :1])],
                              1) # (H*W, 8)

            sample = {'ray_infos': ray_infos,
                      'directions': directions, 
                      'ts': t * torch.ones(len(ray_infos), dtype=torch.long),
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}

            if self.split == 'test_train' and self.perturbation:
                 # append the original (unperturbed) image
                img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, H, W)
                valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
                img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                sample['original_rgbs'] = img
                sample['original_valid_mask'] = valid_mask

        return sample