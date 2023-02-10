import os, glob

from opt import get_opts
import torch
import numpy as np
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

from math import sqrt

# models
from models.nerf import *
from models.rendering import *

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *
import pandas as pd
import pickle
# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
# from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.loggers import WandbLogger
import wandb

import random

import lpips
lpips_alex = lpips.LPIPS(net='alex') # best forward scores

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
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
        self.last_score = {'psnr':0, 'ssim':0, 'lpips':0}
        self.best_score = {'psnr':0, 'ssim':0, 'lpips':0}
    def forward(self, rays, ts, split='train'):
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
                            validation=False if split=='train' else True
                            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage):
        dataset = dataset_dict['phototourism_optimize']
        kwargs = {'root_dir': self.hparams.root_dir}
        if self.hparams.dataset_name == 'phototourism':
            kwargs['img_downscale'] = self.hparams.img_downscale
            kwargs['val_num'] = self.hparams.num_gpus
            kwargs['use_cache'] = self.hparams.use_cache
            kwargs['data_idx'] = self.hparams.data_idx
        elif self.hparams.dataset_name == 'blender':
            kwargs['img_wh'] = tuple(self.hparams.img_wh)
            kwargs['perturbation'] = self.hparams.data_perturb
            kwargs['batch_size'] = self.hparams.batch_size
            kwargs['scale_anneal'] = self.hparams.scale_anneal
            kwargs['min_scale'] = self.hparams.min_scale
            if self.hparams.useNeuralRenderer:
                kwargs['NeuralRenderer_downsampleto'] = (self.hparams.NRDS, self.hparams.NRDS)
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models_to_train)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size, # self.hparams.batch_size a time
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=self.hparams.batch_size, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        rays, ts = batch['rays'], batch['ts']
        rgbs = batch['rgbs']

        results = self(rays, ts, split='train')
        results['rgbs'] = rgbs
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        # self.log('train/loss', loss)
        for k, v in loss_d.items():
            self.log(f'train/{k}', v)
        # self.log('train/psnr', psnr_)
        for k, v in results.items():
            results[k] = v.detach()
        return {"loss": loss, "results": results}

    def training_epoch_end(self, outputs):
        results = defaultdict(list)
        for output in outputs:
            for k,v in output['results'].items():
                results[k] += [v]
        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        rgbs = results['rgbs']
        H, W = self.train_dataset.img_h, self.train_dataset.img_w
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
    
        depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
        self.logger.log_image('viz/train/GT', [img_gt])
        self.logger.log_image('viz/train/pred', [img])
        self.logger.log_image('viz/train/depth', [depth])

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        ssim_ = ssim(img[None,...], img_gt[None,...])

        self.log('train/psnr', psnr_, prog_bar=True)
        self.log('train/ssim', ssim_, prog_bar=True)

    def validation_step(self, batch, batch_nb):
        rays, ts = batch['rays'], batch['ts']
        rgbs =  batch['rgbs']

        results = self(rays, ts, split='val')
        results['rgbs'] = rgbs
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())
        log = {'val_loss': loss}
        for k, v in loss_d.items():
            log[k] = v
        return results, log
    
    def validation_epoch_end(self, outputs):
        results = defaultdict(list)
        log = defaultdict(list)
        for r, l in outputs:
            for k,v in r.items():
                results[k] += [v]
            for k,v in l.items():
                log[k] += [v]
        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        for k, v in log.items():
            log[k] = sum(v)/len(v)
        rgbs = results['rgbs']
        H, W = self.val_dataset.img_h, self.val_dataset.img_w
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
    
        depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
        self.logger.log_image('viz/val/GT', [img_gt])
        self.logger.log_image('viz/val/pred', [img])
        self.logger.log_image('viz/val/depth', [depth])

        psnr_ = psnr(results[f'rgb_{typ}'], rgbs).item()
        ssim_ = ssim(img[None,...], img_gt[None,...]).item()
        lpips_ = lpips_alex((img_gt[None,...]), img[None, ...]).item()
        log['val_psnr'] = psnr_
        log['val_ssim'] = ssim_
        log['val_lpips'] = lpips_
        self.last_score['psnr'] = psnr_
        self.last_score['ssim'] = ssim_
        self.last_score['lpips'] = lpips_
        if self.best_score['psnr'] < psnr_:
            self.best_score['psnr'] = psnr_
        if self.best_score['ssim'] < ssim_:
            self.best_score['ssim'] = ssim_
        if self.best_score['lpips'] > lpips_:
            self.best_score['lpips'] = lpips_
        self.log('val/loss', log['val_loss'])
        self.log('val/psnr', psnr_, prog_bar=True)
        self.log('val/ssim', ssim_, prog_bar=True)
        self.log('val/lpips', lpips_, prog_bar=True)



    # def validation_epoch_end(self, outputs):
    #     if len(outputs) == 1:
    #         global_val.current_epoch = self.current_epoch
    #     else:
    #         global_val.current_epoch = self.current_epoch + 1
    #     mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
    #     mean_ssim = torch.stack([x['val_ssim'] for x in outputs]).mean()
    #     self.log('val/loss', mean_loss)
    #     self.log('val/psnr', mean_psnr, prog_bar=True)
    #     self.log('val/ssim', mean_ssim, prog_bar=True)

    #     if self.hparams.use_mask:
    #         self.log('val/c_l', torch.stack([x['c_l'] for x in outputs]).mean())
    #         self.log('val/f_l', torch.stack([x['f_l'] for x in outputs]).mean())
    #         self.log('val/r_ms', torch.stack([x['r_ms'] for x in outputs]).mean())


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def freeze_weight(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

def main(hparams):
    if hparams.data_idx == -1:
        all_psnr = []
        all_ssim = []
        setup_seed(hparams.seed)
        tsv = glob.glob(os.path.join(hparams.root_dir, '*.tsv'))[0]
        hparams.scene_name = os.path.basename(tsv)[:-4]
        files = pd.read_csv(tsv, sep='\t')
        files = files[~files['id'].isnull()] # remove data without id
        files.reset_index(inplace=True, drop=True)
        with open(os.path.join(hparams.root_dir, f'cache/img_ids.pkl'), 'rb') as f:
                img_ids = pickle.load(f)
        img_ids_test = [id_ for i, id_ in enumerate(img_ids)
                                    if files.loc[i, 'split']=='test']
        
        image_len = len(img_ids_test)
        for data_idx in range(image_len):
            hparams.data_idx = data_idx

            system = NeRFSystem(hparams)
            checkpoint_callback = \
                ModelCheckpoint(dirpath=os.path.join(hparams.save_dir,
                                                    f'optimize_ckpts/{hparams.exp_name}'),
                                save_last=True, 
                                monitor='val/psnr',
                                mode='max',
                                save_top_k=2)
            pbar = TQDMProgressBar(refresh_rate=1)
            callbacks = [checkpoint_callback, pbar]
            if hparams.use_mean_embedding:
                exp_name = hparams.exp_name+f'_idx{hparams.data_idx}_mean_embedding'
            else:
                exp_name = hparams.exp_name+f'_idx{hparams.data_idx}'
            logger = WandbLogger(name=exp_name, project='Reproduce_nerf-w_optimize')
            trainer = Trainer(max_epochs=hparams.num_epochs,
                            log_every_n_steps=10, 
                            callbacks=callbacks,
                            logger=logger,
                            enable_model_summary=False,
                            devices= hparams.num_gpus,
                            accelerator='auto',
                            strategy=DDPPlugin(find_unused_parameters=False) if hparams.num_gpus>1 else None,
                            num_sanity_val_steps=0,
                            benchmark=True,
                            profiler="simple" if hparams.num_gpus==1 else None)

            ckpt = torch.load(os.path.join(hparams.save_dir, f'ckpts/{hparams.exp_name}/last.ckpt'))
            system.load_state_dict(ckpt['state_dict'])
            freeze_weight(system.nerf_coarse)
            freeze_weight(system.nerf_fine)
            system.nerf_fine.encode_transient = False

            # if hparams.use_mean_embedding:
            #     with torch.no_grad():
            #         import pandas as pd
            #         import pickle
            #         tsv = glob.glob(os.path.join(hparams.root_dir, '*.tsv'))[0]
            #         hparams.scene_name = os.path.basename(tsv)[:-4]
            #         files = pd.read_csv(tsv, sep='\t')
            #         files = files[~files['id'].isnull()] # remove data without id
            #         files.reset_index(inplace=True, drop=True)
            #         with open(os.path.join(hparams.root_dir, f'cache/img_ids.pkl'), 'rb') as f:
            #                 img_ids = pickle.load(f)
            #         img_ids_train = [id_ for i, id_ in enumerate(img_ids) 
            #                                     if files.loc[i, 'split']=='train']
            #         img_ids_test = [id_ for i, id_ in enumerate(img_ids)
            #                                     if files.loc[i, 'split']=='test']
            #         system.embedding_a = system.embedding_a.cuda()
            #         import pdb;pdb.set_trace()
            #         embedding = system.embedding_a(torch.Tensor(img_ids_train).unsqueeze(0).long().cuda())
            #         mean_embedding = torch.mean(embedding,1)
            #         system.embedding_a.weight[img_ids_test] = mean_embedding

            trainer.fit(system)
            wandb.finish()
            with open(os.path.join(hparams.save_dir, f'optimize_ckpts/{hparams.exp_name}/score_{data_idx}.pkl'), 'wb') as f:
                pickle.dump(system.last_score, f)
            
            all_psnr.append(system.last_score['psnr'])
            all_ssim.append(system.last_score['ssim'])
        print("PSNR: ", all_psnr)
        print("SSIM: ", all_ssim)
        print("PSNR: ", sum(all_psnr)/len(all_psnr))
        print("SSIM: ", sum(all_ssim)/len(all_psnr))
                
    else:
        system = NeRFSystem(hparams)
        data_idx = hparams.data_idx
        checkpoint_callback = \
            ModelCheckpoint(dirpath=os.path.join(hparams.save_dir,
                                                f'optimize_ckpts/{hparams.exp_name}'),
                            save_last=True, 
                            monitor='val/psnr',
                            mode='max',
                            save_top_k=2)
        pbar = TQDMProgressBar(refresh_rate=1)
        callbacks = [checkpoint_callback, pbar]
        if hparams.use_mean_embedding:
            exp_name = hparams.exp_name+f'_idx{hparams.data_idx}_mean_embedding'
        else:
            exp_name = hparams.exp_name+f'_idx{hparams.data_idx}'
        logger = WandbLogger(name='new_'+exp_name, project='Reproduce_nerf-w_optimize')
        trainer = Trainer(max_epochs=hparams.num_epochs,
                        log_every_n_steps=10, 
                        callbacks=callbacks,
                        logger=logger,
                        enable_model_summary=False,
                        devices= hparams.num_gpus,
                        accelerator='auto',
                        check_val_every_n_epoch=hparams.val_epoch, 
                        strategy=DDPPlugin(find_unused_parameters=False) if hparams.num_gpus>1 else None,
                        num_sanity_val_steps=0,
                        benchmark=True,
                        profiler="simple" if hparams.num_gpus==1 else None)

        ckpt = torch.load(os.path.join(hparams.save_dir, f'ckpts/{hparams.exp_name}/last.ckpt'))
        system.load_state_dict(ckpt['state_dict'])
        freeze_weight(system.nerf_coarse)
        freeze_weight(system.nerf_fine)
        system.nerf_fine.encode_transient = False

        if hparams.use_mean_embedding:
            with torch.no_grad():
                tsv = glob.glob(os.path.join(hparams.root_dir, '*.tsv'))[0]
                hparams.scene_name = os.path.basename(tsv)[:-4]
                files = pd.read_csv(tsv, sep='\t')
                files = files[~files['id'].isnull()] # remove data without id
                files.reset_index(inplace=True, drop=True)
                with open(os.path.join(hparams.root_dir, f'cache/img_ids.pkl'), 'rb') as f:
                        img_ids = pickle.load(f)
                img_ids_train = [id_ for i, id_ in enumerate(img_ids) 
                                            if files.loc[i, 'split']=='train']
                img_ids_test = [id_ for i, id_ in enumerate(img_ids)
                                            if files.loc[i, 'split']=='test']
                system.embedding_a = system.embedding_a.cuda()
                embedding = system.embedding_a(torch.Tensor(img_ids_train).unsqueeze(0).long().cuda())
                mean_embedding = torch.mean(embedding,1)
                system.embedding_a.weight[img_ids_test] = mean_embedding

        trainer.fit(system)
        with open(os.path.join(hparams.save_dir, f'optimize_ckpts/{hparams.exp_name}/score_{data_idx}.pkl'), 'wb') as f:
                pickle.dump(system.last_score, f)
        torch.save(system.embedding_a(system.train_dataset[0]['ts']).detach().cpu(), hparams.save_dir+'/'+f'optimize_ckpts/{hparams.exp_name}/embedding_{data_idx}.pt')

if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)