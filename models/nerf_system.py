from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule

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
import camera
from datasets.ray_utils import get_rays

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)

        self.loss = loss_dict['nerfw'](coef=1)

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

        self.nerf_coarse = NeRF('coarse',
                                xyz_L=hparams['nerf.N_emb_xyz'],
                                dir_L=hparams['nerf.N_emb_dir'],
                                barf_c2f=hparams['barf.c2f'])
        self.models = {'coarse': self.nerf_coarse}
        if hparams['nerf.N_importance'] > 0:
            self.nerf_fine = NeRF('fine',
                                  xyz_L=hparams['nerf.N_emb_xyz'],
                                  dir_L=hparams['nerf.N_emb_dir'],
                                  encode_appearance=hparams['nerf.encode_a'],
                                  in_channels_a=hparams['nerf.a_dim'],
                                  encode_transient=hparams['nerf.encode_t'],
                                  in_channels_t=hparams['nerf.t_dim'],
                                  beta_min=hparams['nerf.beta_min'],
                                  barf_c2f=hparams['barf.c2f'])
            self.models['fine'] = self.nerf_fine
        self.models_to_train += [self.models]

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays, ts, train=True):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        chunk= 1024*32
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
                            chunk, # chunk size is effective in val mode
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
            kwargs['val_num'] = 2 #self.hparams['num_gpus']
            kwargs['use_cache'] = self.hparams['phototourism.use_cache']
            kwargs['fewshot'] = self.hparams['phototourism.fewshot']
            kwargs['N_vocab'] = self.hparams['N_vocab']
        elif self.hparams['dataset_name'] == 'blender':
            kwargs['img_wh'] = self.hparams['blender.img_wh']
            kwargs['perturbation'] = self.hparams['blender.data_perturb']

        if self.hparams['barf.camera.noise'] == -1:
            self.train_dataset = dataset(split='train', camera_noise=-1, **kwargs)
            self.val_dataset = dataset(split='val', camera_noise=-1 , **kwargs)
        else:
            self.train_dataset = dataset(split='train', camera_noise=self.hparams['barf.camera.noise'], **kwargs)
            self.val_dataset = dataset(split='val', camera_noise=self.train_dataset.pose_noises , **kwargs)
        self.build_pose_networks()

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams['optimizer.type'], self.hparams['optimizer.lr'], self.models_to_train)
        scheduler_nerf = get_scheduler(self.hparams['optimizer.scheduler.type'], self.hparams['optimizer.lr'], self.hparams['optimizer.scheduler.lr_end'], self.hparams['max_steps'], self.optimizer)
        optimizer = [self.optimizer]
        scheduler = [{'scheduler':scheduler_nerf, 'interval':'step'}]
        
        if self.hparams['barf.refine']:
            self.optimizer_pose = get_optimizer(self.hparams['optimizer_pose.type'], self.hparams['optimizer_pose.lr'], self.se3_refine)
            scheduler_pose = get_scheduler(self.hparams['optimizer_pose.scheduler.type'], self.hparams['optimizer_pose.lr'], self.hparams['optimizer_pose.scheduler.lr_end'], self.hparams['max_steps'], self.optimizer_pose)
            optimizer += [self.optimizer_pose]
            scheduler += [{'scheduler':scheduler_pose, 'interval':'step'}]
        
        return optimizer, scheduler
        # return [self.optimizer], [scheduler]

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
        ts_idx = batch['ts_idx'] if 'ts_idx' in batch.keys() else ts

        if self.hparams['barf.refine']:
            pose_refine = camera.lie.se3_to_SE3(self.se3_refine(ts_idx))
            refined_pose = camera.pose.compose([pose_refine, pose])
            rays_o, rays_d = get_rays(directions, refined_pose) # both (h*w, 3)
        else:
            rays_o, rays_d = get_rays(directions, pose) # both (h*w, 3)
        rays = torch.cat([rays_o, rays_d, ray_infos], 1)
        
        results = self(rays, ts)
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())
        
        if self.hparams['barf.refine']:
            self.optimizers()[0].zero_grad()
            self.optimizers()[1].zero_grad()
            self.manual_backward(loss)
            self.optimizers()[0].step()
            self.lr_schedulers()[0].step()
            self.optimizers()[1].step()
            self.lr_schedulers()[1].step()
        else:
            self.optimizers().zero_grad()
            self.manual_backward(loss)
            self.optimizers().step()
            self.lr_schedulers().step()
            
        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        if self.hparams['barf.refine']:
            self.log('lr_pose', get_learning_rate(self.optimizers()[1]))
        self.log('train/loss', loss)
        for k, v in loss_d.items():
            self.log(f'train/{k}', v, prog_bar=True)
        self.log('train/psnr', psnr_, prog_bar=True)
        
        # barf
        self.nerf_coarse.progress.data.fill_(self.global_step/self.hparams['max_steps'])
        self.nerf_fine.progress.data.fill_(self.global_step/self.hparams['max_steps'])

        return loss

    def validation_step(self, batch, batch_nb):
        ray_infos, rgbs, ts, directions, pose = batch['ray_infos'], batch['rgbs'], batch['ts'], batch['directions'], batch['c2w']
        ts_idx = batch['ts_idx'] if 'ts_idx' in batch.keys() else ts
        ray_infos = ray_infos.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        ts = ts.squeeze() # (H*W)
        ts_idx = ts_idx.squeeze()
        directions = directions.squeeze()
        if self.hparams['barf.refine']:
            pose_refine = camera.lie.se3_to_SE3(self.se3_refine(ts_idx))
            refined_pose = camera.pose.compose([pose_refine, pose])
            rays_o, rays_d = get_rays(directions, refined_pose) # both (h*w, 3)
        else:
            rays_o, rays_d = get_rays(directions, pose.squeeze()) # both (h*w, 3)
        rays = torch.cat([rays_o, rays_d, ray_infos], 1)
        
        results = self(rays, ts, train=False)
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())
        log = {'val_loss': loss}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_nb == 0:
            if self.hparams['dataset_name'] == 'phototourism':
                WH = batch['img_wh']
                W, H = WH[0, 0].item(), WH[0, 1].item()
            else:
                W, H = self.hparams['blender.img_wh']
            img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            img_coarse = results[f'rgb_coarse'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            if 'rgb_fine_static' not in results.keys():
                img_pred = results['rgb_fine'].view(H,W,3).permute(2,0,1).cpu()
                depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
                self.logger.log_image('val/viz/GT', [img_gt])
                self.logger.log_image('val/viz/pred', [img_pred])
                self.logger.log_image('val/viz/pred_static', [img_pred])
                self.logger.log_image('val/viz/depth', [depth])
            else:
                img_static = results['rgb_fine_static'].view(H,W,3).permute(2,0,1).cpu()
                img_pred_transient = results['_rgb_fine_transient'].view(H,W,3).permute(2,0,1).cpu()
                beta = results['beta'].view(H,W).cpu()
                depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
                self.logger.log_image('val/viz/GT', [img_gt])
                self.logger.log_image('val/viz/pred', [img])
                self.logger.log_image('val/viz/pred_coarse', [img_coarse])
                self.logger.log_image('val/viz/depth', [depth])
                self.logger.log_image('val/viz/pred_static', [img_static])
                self.logger.log_image('val/viz/pred_transient', [img_pred_transient])
                self.logger.log_image('val/viz/beta', [beta])

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        log['val_psnr'] = psnr_
        if 'rgb_fine_static' in results.keys():
            psnr_static = psnr(results['rgb_fine_static'], rgbs)
            log['val_static_psnr'] = psnr_static

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)
        
        if 'val_static_psnr' in outputs[0]:
            mean_static_psnr = torch.stack([x['val_static_psnr'] for x in outputs]).mean()
            self.log('val/static_psnr', mean_static_psnr, prog_bar=True)

    def build_pose_networks(self):
        self.se3_refine = torch.nn.Embedding(self.train_dataset.N_images_train,6).to('cuda')
        torch.nn.init.zeros_(self.se3_refine.weight)