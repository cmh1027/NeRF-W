import sys
import os
import random
import argparse
import numpy as np
import torch

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


from configs.config import parse_args
from models.nerf_system import NeRFSystem


parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Path to config file.", required=False, default='./configs/phototourism.yaml')
parser.add_argument("opts", nargs=argparse.REMAINDER,
                    help="Modify hparams. Example: train.py resume out_dir TRAIN.BATCH_SIZE 2")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(hparams):
    assert (hparams['barf.refine']==True) or (hparams['barf.refine']==False and hparams['barf.c2f']==None), \
        "if you don't refine poses, barf.c2f must be None"
    
    setup_seed(hparams['seed'])
    system = NeRFSystem(hparams)
    checkpoint_callback = \
        ModelCheckpoint(dirpath=os.path.join(hparams['out_dir'],
                                              'ckpts', hparams['exp_name']),
                        save_last=True, 
                        monitor='val/psnr',
                        mode='max',
                        save_top_k=2)
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [checkpoint_callback, pbar]
    logger = WandbLogger(name=hparams['exp_name'], project='pose_refine_nerfw')

    max_steps = hparams['max_steps']*2 if hparams['barf.refine'] else hparams['max_steps']
    trainer = Trainer(max_steps=max_steps, 
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      val_check_interval=hparams['val.check_interval'], 
                      devices= hparams['num_gpus'],
                      accelerator='auto',
                      strategy='dp' if hparams['num_gpus'] > 1 else None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if hparams['num_gpus']==1 else None)

    trainer.fit(system)


if __name__ == '__main__':
    main(parse_args(parser))