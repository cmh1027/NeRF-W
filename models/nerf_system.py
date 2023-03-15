from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
import cv2
# models
from models.nerf import *
from models.rendering import *
# optimizer, scheduler, visualization
from utils import *
# losses
from losses import loss_dict
# metrics
from metrics import *
from datasets import dataset_dict
# barf
import camera, os
from datasets.ray_utils import get_rays


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)

        self.loss = loss_dict['nerfw'](coef=1, color_optim=hparams['color_optim'], feat_optim=hparams['feat_optim'])

        self.models_to_train = []
        self.embedding_xyz = torch.nn.Identity()
        self.embedding_dir = torch.nn.Identity()
        # self.embedding_xyz = PosEmbedding(hparams['nerf.N_emb_xyz']-1, hparams['nerf.N_emb_xyz'])
        # self.embedding_dir = PosEmbedding(hparams['nerf.N_emb_dir']-1, hparams['nerf.N_emb_dir'])
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}
        self.automatic_optimization = False

        if hparams['nerf.encode_a']:
            self.embedding_a = torch.nn.Embedding(hparams['N_vocab'], hparams['nerf.a_dim'])
            self.embeddings['a'] = self.embedding_a
            self.models_to_train += [self.embedding_a]
        if hparams['nerf.encode_t']:
            self.embedding_t = torch.nn.Embedding(hparams['N_vocab'], hparams['nerf.t_dim'])
            self.embeddings['t'] = self.embedding_t
            self.models_to_train += [self.embedding_t]

        if hparams['nerf.N_importance'] > 0:
            self.nerf_coarse = NeRF('coarse',
                                    xyz_L=hparams['nerf.N_emb_xyz'],
                                    dir_L=hparams['nerf.N_emb_dir'],
                                    barf_c2f=hparams['barf.c2f'],
                                    activation=hparams['activation'])
            self.models = {'coarse': self.nerf_coarse}
            self.nerf_fine = NeRF('fine',
                                  xyz_L=hparams['nerf.N_emb_xyz'],
                                  dir_L=hparams['nerf.N_emb_dir'],
                                  encode_appearance=hparams['nerf.encode_a'],
                                  in_channels_a=hparams['nerf.a_dim'],
                                  encode_transient=hparams['nerf.encode_t'],
                                  in_channels_t=hparams['nerf.t_dim'],
                                  beta_min=hparams['nerf.beta_min'],
                                  barf_c2f=hparams['barf.c2f'],
                                  activation=hparams['activation'],
                                  encode_transient_front=hparams['nerf.encode_t_front'])
            self.models['fine'] = self.nerf_fine
        else:
            self.nerf_coarse = NeRF('coarse',
                                    xyz_L=hparams['nerf.N_emb_xyz'],
                                    dir_L=hparams['nerf.N_emb_dir'],
                                    encode_appearance=hparams['nerf.encode_a'],
                                  in_channels_a=hparams['nerf.a_dim'],
                                  encode_transient=hparams['nerf.encode_t'],
                                  in_channels_t=hparams['nerf.t_dim'],
                                  beta_min=hparams['nerf.beta_min'],
                                  barf_c2f=hparams['barf.c2f'],
                                  activation=hparams['nerf.activation'],
                                  encode_transient_front=hparams['nerf.encode_t_front'])
            self.models = {'coarse': self.nerf_coarse}
        self.models_to_train += [self.models]



    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays, ts, train=True):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        chunk= 1024*16
        for i in range(0, B, chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+chunk],
                            ts[i:i+chunk],
                            self.hparams['nerf.N_samples'],
                            self.hparams['nerf.use_disp'],
                            self.hparams['nerf.perturb'],
                            self.hparams['nerf.noise_std'],
                            self.hparams['nerf.N_importance'],
                            1024*64*4, # chunk size is effective in val mode
                            self.train_dataset.white_back,
                            validation=False if train else True
                            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams['dataset_name']]
        kwargs = {'root_dir': self.hparams['root_dir']}
        if self.hparams['dataset_name'] == 'phototourism':
            kwargs['img_downscale'] = self.hparams['phototourism.img_downscale']
            kwargs['use_cache'] = self.hparams['phototourism.use_cache']
            kwargs['fewshot'] = self.hparams['phototourism.fewshot']
            kwargs['N_vocab'] = self.hparams['N_vocab']
            kwargs['feat_dir'] = self.hparams['feat_dir'] if self.hparams['feat_optim'] else None
            kwargs['pca_info_dir'] = self.hparams['pca_info_dir'] if self.hparams['feat_optim'] else None
            
        elif self.hparams['dataset_name'] == 'blender':
            kwargs['img_wh'] = self.hparams['blender.img_wh']
            kwargs['perturbation'] = self.hparams['blender.data_perturb']
            kwargs['feat_dir'] = self.hparams['feat_dir'] if self.hparams['feat_optim'] else None
            kwargs['pca_info_dir'] = self.hparams['pca_info_dir'] if self.hparams['feat_optim'] else None

        if self.hparams['barf.camera.noise'] == -1:
            self.train_dataset = dataset(split='train', camera_noise=-1, **kwargs)
            self.val_dataset = dataset(split='val', camera_noise=-1, img_idx=self.hparams['val.img_idx'], **kwargs)
        else:
            self.train_dataset = dataset(split='train', camera_noise=self.hparams['barf.camera.noise'], **kwargs)
            self.val_dataset = dataset(split='val', camera_noise=self.hparams['barf.camera.noise'], img_idx=self.hparams['val.img_idx'], **kwargs)
        self.build_pose_networks()
        if self.hparams['dataset_name'] == 'phototourism':
            gt_poses = torch.stack([torch.from_numpy(self.train_dataset.GT_poses_dict[i]) for i in self.train_dataset.img_ids_train])
        elif self.hparams['dataset_name'] == 'blender':
            gt_poses = self.train_dataset.gt_poses # (N, 3)
        gt_min, _ = gt_poses.min(dim=0)
        gt_max, _ = gt_poses.max(dim=0)
        self.max_dist = torch.sqrt(((gt_min - gt_max) ** 2).sum()).item()

    def configure_optimizers(self):
        self.model_optimizer = get_optimizer(self.hparams['optimizer.type'], self.hparams['optimizer.lr'], self.models_to_train)
        self.model_scheduler = get_scheduler(self.hparams['optimizer.scheduler.type'], self.hparams['optimizer.lr'], self.hparams['optimizer.scheduler.lr_end'], self.hparams['max_steps'], self.model_optimizer)
        optimizer = [self.model_optimizer]
        scheduler = [{'scheduler':self.model_scheduler, 'interval':'step'}]
        
        if self.hparams['barf.refine']:
            self.pose_optimizer = get_optimizer(self.hparams['optimizer_pose.type'], self.hparams['optimizer_pose.lr'], self.se3_refine)
            self.pose_scheduler = get_scheduler(self.hparams['optimizer_pose.scheduler.type'], self.hparams['optimizer_pose.lr'], self.hparams['optimizer_pose.scheduler.lr_end'], self.hparams['max_steps'], self.pose_optimizer)
            optimizer += [self.pose_optimizer]
            scheduler += [{'scheduler':self.pose_scheduler, 'interval':'step'}]
        
        return optimizer, scheduler

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=8,
                          batch_size=self.hparams['train.batch_size'],
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        ray_infos, rgbs, ts, directions, pose = batch['ray_infos'], batch['rgbs'], batch['ts'], batch['directions'], batch['c2w']
        feats = None 
        if self.hparams['feat_optim']:
            feats = batch['feats']
        ts_idx = batch['ts_idx'] if 'ts_idx' in batch.keys() else ts

        if self.hparams['barf.refine']:
            pose_refine = camera.lie.se3_to_SE3(self.se3_refine(ts_idx))
            refined_pose = camera.pose.compose([pose_refine, pose])
            rays_o, rays_d = get_rays(directions, refined_pose) # both (h*w, 3)
        else:
            rays_o, rays_d = get_rays(directions, pose) # both (h*w, 3)
        rays = torch.cat([rays_o, rays_d, ray_infos], 1)
        
        results = self(rays, ts)
        loss_d = self.loss(results, rgbs, feats)
        loss = sum(l for l in loss_d.values())
        
        if self.hparams['barf.refine']:
            self.model_optimizer.zero_grad()
            self.pose_optimizer.zero_grad()
            self.manual_backward(loss)
            self.model_optimizer.step()
            self.model_scheduler.step()
            self.pose_optimizer.step()
            self.pose_scheduler.step()
        else:
            self.model_optimizer.zero_grad()
            self.manual_backward(loss)
            self.model_optimizer.step()
            self.model_scheduler.step()
            
        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        self.log('lr', get_learning_rate(self.model_optimizer))
        if self.hparams['barf.refine']:
            self.log('lr_pose', get_learning_rate(self.pose_optimizer))
        self.log('train/loss', loss)
        for k, v in loss_d.items():
            self.log(f'train/{k}', v, prog_bar=True)
        self.log('train/psnr', psnr_, prog_bar=True)
        
        # barf
        if self.hparams['barf.refine']:
            self.nerf_coarse.progress.data.fill_(self.global_step/(self.hparams['max_steps']*2))
            if self.hparams['nerf.N_importance'] > 0:
                self.nerf_fine.progress.data.fill_(self.global_step/(self.hparams['max_steps']*2))

        return loss

    def validation_step(self, batch, batch_nb):
        ray_infos, rgbs, ts, directions, pose = batch['ray_infos'], batch['rgbs'], batch['ts'], batch['directions'], batch['c2w']
        
        feats = None
        if self.hparams['feat_optim']:
            feats, m, c = batch['feats'][0], batch['m'][0], batch['c'][0]
        
        ts_idx = batch['ts_idx'] if 'ts_idx' in batch.keys() else ts
        ray_infos = ray_infos.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        ts = ts.squeeze() # (H*W)
        ts_idx = ts_idx.squeeze()
        directions = directions.squeeze()
        
        ### get refined pose
        if self.hparams['barf.refine']:
            pose_refine = camera.lie.se3_to_SE3(self.se3_refine(ts_idx))
            refined_pose = camera.pose.compose([pose_refine, pose])
            rays_o, rays_d = get_rays(directions, refined_pose) # both (h*w, 3)
        else:
            rays_o, rays_d = get_rays(directions, pose.squeeze()) # both (h*w, 3)
        rays = torch.cat([rays_o, rays_d, ray_infos], 1)
        
        ### forward
        results = self(rays, ts, train=False)
        loss_d = self.loss(results, rgbs, feats)
        loss = sum(l for l in loss_d.values())
        log = {'val_loss': loss}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        log['val_psnr'] = psnr_
    
        ### log image
        idx = self.val_dataset.img_idx[batch_nb]
        if self.hparams['dataset_name'] == 'phototourism':
            WH = batch['img_wh']
            W, H = WH[0, 0].item(), WH[0, 1].item()
        else:
            W, H = self.hparams['blender.img_wh']
        
        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
        self.logger.log_image(f'val_{idx}/viz/GT', [img_gt])
        if feats is not None:
            gt_pca = get_pca_img(feats, m, c).permute(2,0,1)
            self.logger.log_image(f'val_{idx}/viz/GT_feat', [gt_pca])
        
        
        for log_img_name in self.hparams['val.log_image_list']:
            try:
                if 'depth' in log_img_name:
                    depth = results[log_img_name].view(H, W).cpu() # (3, H, W)
                    img = visualize_depth(depth)
                elif 'feat' in log_img_name and feats is not None:
                    feat_map = results[log_img_name]
                    img = get_pca_img(feat_map.view(feats.shape), m, c).permute(2,0,1)
                else:
                    img = results[log_img_name].view(H, W, -1).permute(2, 0, 1).cpu() # (3, H, W)
                self.logger.log_image(f'val_{idx}/viz/{log_img_name}', [img])
            except:
                pass

        return log
    
    @torch.no_grad()
    def training_epoch_end(self, outputs):
        if self.hparams['log.poses'] is False:
            return
        if self.hparams['dataset_name'] == 'phototourism':
            noised_poses = torch.stack([self.train_dataset.poses_dict[i] for i in self.train_dataset.img_ids_train])
            gt_poses = torch.stack([torch.from_numpy(self.train_dataset.GT_poses_dict[i]) for i in self.train_dataset.img_ids_train])
        elif self.hparams['dataset_name'] == 'blender':
            noised_poses = self.train_dataset.poses
            gt_poses = self.train_dataset.gt_poses

        pose_refine_ = camera.lie.se3_to_SE3(self.se3_refine.weight).cpu() # (N, 3, 4)
        refine_poses = camera.pose.compose([pose_refine_, noised_poses])
        campos_gt, campos_pred = gt_poses[..., 3], refine_poses[..., 3] # (N, 3)
        dist_gt = distance_mat(campos_gt, campos_gt, normalize=True, c=self.max_dist) # (N, N)
        dist_pred = distance_mat(campos_pred, campos_pred, normalize=True, c=self.max_dist)
        eps = 1e-7
        group_loss = torch.abs(torch.log((dist_gt + eps) / (dist_pred + eps))).sum().item()
        self.log('train/group_loss', group_loss)

        if self.hparams["nerf.encode_a"] is True:
            appearance = self.embedding_a.weight[:len(dist_gt)] # (N, d)
            sim = similarity_mat(appearance, appearance, normalize=True).to(dist_gt.device) # (N, N)
            sim_loss = torch.abs(torch.log((dist_gt + eps) / (1 - sim + eps))).sum().item()
            self.log('train/a_sim_loss', sim_loss)

        if self.hparams["nerf.encode_t"] is True:
            transient = self.embedding_t.weight[:len(dist_gt)] # (N, d)
            sim = similarity_mat(transient, transient, normalize=True).to(dist_gt.device) # (N, N)
            sim_loss = torch.abs(torch.log((dist_gt + eps) / (1 - sim + eps))).sum().item()
            self.log('train/t_sim_loss', sim_loss)

        gt_poses = torch.stack([parse_raw_camera(p) for p in gt_poses.float()],dim=0)
        refine_poses = torch.stack([parse_raw_camera(p) for p in refine_poses.float()],dim=0)
        
        np.save(os.path.join(self.hparams["pose_path"], f'refined_pose{self.current_epoch}.npy'), refine_poses.cpu().numpy())
        if self.current_epoch == 0:
            np.save(os.path.join(self.hparams["pose_path"], f'gt_pose.npy'), gt_poses.cpu().numpy())
        if self.hparams["nerf.encode_t"] is True:
            np.save(os.path.join(self.hparams["pose_path"], f't_embedding{self.current_epoch}.npy'), self.embedding_t.weight.detach().cpu().numpy())
        if self.hparams["nerf.encode_a"] is True:
            np.save(os.path.join(self.hparams["pose_path"], f'a_embedding{self.current_epoch}.npy'), self.embedding_a.weight.detach().cpu().numpy())
            
        
        # if self.current_epoch == 0:
        #     path = os.path.join(self.hparams["pose_path"], 'campos_gt.png')
        #     draw_pos(campos_gt[:, 0].cpu().numpy(), campos_gt[:, 1].cpu().numpy(), campos_gt[:, 2].cpu().numpy(), path)
        #     img = cv2.imread(path)
        #     self.logger.log_image("train/campos_gt", [img])
        # path = os.path.join(self.hparams["pose_path"], 'campos_pred.png')
        # draw_pos(campos_pred[:, 0].cpu().numpy(), campos_pred[:, 1].cpu().numpy(), campos_pred[:, 2].cpu().numpy(), path)
        # img = cv2.imread(path)
        # self.logger.log_image("train/campos_pred", [img])
        
        if self.current_epoch == 0:
            path = os.path.join(self.hparams["pose_path"], 'gt_pose.png')
            generate_videos_pose(path, pose=None, pose_ref=gt_poses, sample_nums=100, cam_depth=0.5, scaling=True, connect=False)
            self.logger.log_image("train/gt_pose", [cv2.imread(path)])
        path = os.path.join(self.hparams["pose_path"], 'pose_noscale.png')
        generate_videos_pose(path, pose=refine_poses, pose_ref=gt_poses, sample_nums=100, cam_depth=0.5, scaling=False, connect=False)
        self.logger.log_image("train/pose_noscale", [cv2.imread(path)])
        path = os.path.join(self.hparams["pose_path"], 'pose_scale.png')
        generate_videos_pose(path, pose=refine_poses, pose_ref=gt_poses, sample_nums=100, cam_depth=0.5, scaling=True, connect=False)
        self.logger.log_image("train/pose_scale", [cv2.imread(path)])
        path = os.path.join(self.hparams["pose_path"], 'pose_noscale_connect.png')
        generate_videos_pose(path, pose=refine_poses, pose_ref=gt_poses, sample_nums=100, cam_depth=0.5, scaling=False, connect=True)
        self.logger.log_image("train/pose_noscale_connect", [cv2.imread(path)])
        path = os.path.join(self.hparams["pose_path"], 'pose_scale_connect.png')
        generate_videos_pose(path, pose=refine_poses, pose_ref=gt_poses, sample_nums=100, cam_depth=0.5, scaling=True, connect=True)
        self.logger.log_image("train/pose_scale_connect", [cv2.imread(path)])


    def validation_epoch_end(self, outputs):
        if self.hparams['dataset_name'] == 'phototourism':
            noised_poses = torch.stack([self.val_dataset.poses_dict[i] for i in self.val_dataset.img_ids_train])
            gt_poses = torch.stack([torch.from_numpy(self.val_dataset.GT_poses_dict[i]) for i in self.val_dataset.img_ids_train])
        elif self.hparams['dataset_name'] == 'blender':
            noised_poses = self.val_dataset.poses
            gt_poses = self.val_dataset.gt_poses
        pose_refine_ = camera.lie.se3_to_SE3(self.se3_refine.weight).cpu()
        refine_poses = camera.pose.compose([pose_refine_,noised_poses])
        try:
            pose_error = pose_metric(refine_poses, gt_poses)
        except:
            pose_error = None
            print("pose alignment is not converged")

        
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        self.log('val/loss', mean_loss)
        if pose_error is not None:
            self.log('val/pose_R', pose_error['R'].mean())
            self.log('val/pose_t', pose_error['t'].mean())
        self.log('val/psnr', mean_psnr, prog_bar=True)
        

    def build_pose_networks(self):
        self.se3_refine = torch.nn.Embedding(self.train_dataset.N_images_train,6).to('cuda')
        torch.nn.init.zeros_(self.se3_refine.weight)