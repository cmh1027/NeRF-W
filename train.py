import os
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import *
from models.rendering import *

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # self.hparams = hparams
        self.save_hyperparameters(hparams)

        self.loss = loss_dict['nerfw'](coef=1)

        self.models_to_train = []
        self.embedding_xyz = PosEmbedding(hparams.N_emb_xyz-1, hparams.N_emb_xyz)
        self.embedding_dir = PosEmbedding(hparams.N_emb_dir-1, hparams.N_emb_dir)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        if hparams.encode_a:
            self.embedding_a = torch.nn.Embedding(hparams.N_vocab, hparams.N_a)
            self.embeddings['a'] = self.embedding_a
            self.models_to_train += [self.embedding_a]
        if hparams.encode_t:
            self.embedding_t = torch.nn.Embedding(hparams.N_vocab, hparams.N_tau)
            self.embeddings['t'] = self.embedding_t
            self.models_to_train += [self.embedding_t]

        self.nerf_coarse = NeRF('coarse',
                                in_channels_xyz=6*hparams.N_emb_xyz+3,
                                in_channels_dir=6*hparams.N_emb_dir+3)
        self.models = {'coarse': self.nerf_coarse}
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF('fine',
                                  in_channels_xyz=6*hparams.N_emb_xyz+3,
                                  in_channels_dir=6*hparams.N_emb_dir+3,
                                  encode_appearance=hparams.encode_a,
                                  in_channels_a=hparams.N_a,
                                  encode_transient=hparams.encode_t,
                                  in_channels_t=hparams.N_tau,
                                  beta_min=hparams.beta_min)
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
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            ts[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back,
                            validation=False if train else True
                            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir}
        if self.hparams.dataset_name == 'phototourism':
            kwargs['img_downscale'] = self.hparams.img_downscale
            kwargs['val_num'] = self.hparams.num_gpus
            kwargs['use_cache'] = self.hparams.use_cache
            kwargs['fewshot'] = self.hparams.fewshot
        elif self.hparams.dataset_name == 'blender':
            kwargs['img_wh'] = tuple(self.hparams.img_wh)
            kwargs['perturbation'] = self.hparams.data_perturb
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models_to_train)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [{'scheduler':scheduler, 'interval':'step'}]
        # return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        rays, rgbs, ts = batch['rays'], batch['rgbs'], batch['ts']
        results = self(rays, ts)
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())

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
            if self.hparams.dataset_name == 'phototourism':
                WH = batch['img_wh']
                W, H = WH[0, 0].item(), WH[0, 1].item()
            else:
                W, H = self.hparams.img_wh
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


def main(hparams):
    setup_seed(hparams.seed)
    system = NeRFSystem(hparams)
    checkpoint_callback = \
        ModelCheckpoint(dirpath=os.path.join(hparams.save_dir,
                                              f'ckpts/{hparams.exp_name}'),
                        save_last=True, 
                        monitor='val/psnr',
                        mode='max',
                        save_top_k=2)
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [checkpoint_callback, pbar]
    logger = WandbLogger(name=hparams.exp_name, project='pose_refine_nerfw')

    trainer = Trainer(max_steps=hparams.num_steps, 
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      val_check_interval=hparams.val_check_interval, 
                      devices= hparams.num_gpus,
                      accelerator='auto',
                      strategy=DDPPlugin(find_unused_parameters=False) if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus==1 else None)

    trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)