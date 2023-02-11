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

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)

        self.loss = loss_dict['nerfw'](coef=1)

        self.models_to_train = []
        self.embedding_xyz = PosEmbedding(hparams['nerf.N_emb_xyz']-1, hparams['nerf.N_emb_xyz'])
        self.embedding_dir = PosEmbedding(hparams['nerf.N_emb_dir']-1, hparams['nerf.N_emb_dir'])
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
                                in_channels_xyz=6*hparams['nerf.N_emb_xyz']+3,
                                in_channels_dir=6*hparams['nerf.N_emb_dir']+3)
        self.models = {'coarse': self.nerf_coarse}
        if hparams['nerf.N_importance'] > 0:
            self.nerf_fine = NeRF('fine',
                                  in_channels_xyz=6*hparams['nerf.N_emb_xyz']+3,
                                  in_channels_dir=6*hparams['nerf.N_emb_dir']+3,
                                  encode_appearance=hparams['nerf.encode_a'],
                                  in_channels_a=hparams['nerf.a_dim'],
                                  encode_transient=hparams['nerf.encode_t'],
                                  in_channels_t=hparams['nerf.t_dim'],
                                  beta_min=hparams['nerf.beta_min'])
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
            kwargs['val_num'] = self.hparams['num_gpus']
            kwargs['use_cache'] = self.hparams['phototourism.use_cache']
            kwargs['fewshot'] = self.hparams['phototourism.fewshot']
        elif self.hparams['dataset_name'] == 'blender':
            kwargs['img_wh'] = self.hparams['blender.img_wh']
            kwargs['perturbation'] = self.hparams['blender.data_perturb']
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams['optimizer.type'], self.hparams['optimizer.lr'], self.models_to_train)
        scheduler = get_scheduler(self.hparams['optimizer.scheduler.type'], self.hparams['optimizer.lr'], self.hparams['optimizer.scheduler.lr_end'], self.hparams['max_steps'], self.optimizer)
        
        self.optimizer_pose = get_optimizer(self.hparams['optimizer_pose.type'], self.hparams['optimizer_pose.lr'], self.models_to_train)
        scheduler_pose = get_scheduler(self.hparams['optimizer_pose.scheduler.type'], self.hparams['optimizer_pose.lr'], self.hparams['optimizer_pose.scheduler.lr_end'], self.hparams['max_steps'], self.optimizer_pose)
        
        return [self.optimizer, self.optimizer_pose], [{'scheduler':scheduler, 'interval':'step'}, {'scheduler':scheduler_pose, 'interval':'step'}]
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
        (optim_nerf, optim_pose) = self.optimizers()
        (schedul_nerf, schedul_pose) = self.lr_schedulers()
        
        rays, rgbs, ts = batch['rays'], batch['rgbs'], batch['ts']
        results = self(rays, ts)
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())
        
        optim_nerf.zero_grad()
        self.manual_backward(loss)
        optim_nerf.step()
        schedul_nerf.step()

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)
        for k, v in loss_d.items():
            self.log(f'train/{k}', v, prog_bar=True)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        rays, rgbs, ts = batch['rays'], batch['rgbs'], batch['ts']
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        ts = ts.squeeze() # (H*W)
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
            img_static = results['rgb_fine_static'].view(H,W,3).permute(2,0,1).cpu()
            img_pred_transient = results['_rgb_fine_transient'].view(H,W,3).permute(2,0,1).cpu()
            beta = results['beta'].view(H,W).cpu()
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            self.logger.log_image('viz/val/GT', [img_gt])
            self.logger.log_image('viz/val/pred', [img])
            self.logger.log_image('viz/val/pred_coarse', [img])
            self.logger.log_image('viz/val/depth', [depth])
            self.logger.log_image('viz/pred_static', [img_static])
            self.logger.log_image('viz/pred_transient', [img_pred_transient])
            self.logger.log_image('viz/beta', [beta])

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        psnr_static = psnr(results['rgb_fine_static'], rgbs)
        log['val_psnr'] = psnr_
        log['val_static_psnr'] = psnr_static

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        mean_static_psnr = torch.stack([x['val_static_psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)
        self.log('val/static_psnr', mean_static_psnr, prog_bar=True)
