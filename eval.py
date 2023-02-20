import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays
from models.nerf import *

from utils import load_ckpt
from utils import visualize_depth

from datasets import PhototourismDataset
from datasets.ray_utils import get_rays
# from datasets import dataset_dict
# from datasets.depth_utils import *

torch.backends.cudnn.benchmark = True

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/hub_data2/injae/nerf/phototourism/brandenburg_gate',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='phototourism',
                        choices=['blender', 'phototourism'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test', 'test_train', 'video'])
    parser.add_argument('--img_wh', nargs="+", type=int, default=[400, 400],
                        help='resolution (img_w, img_h) of the image')
    # for phototourism
    parser.add_argument('--img_downscale', type=int, default=2,
                        help='how much to downscale the images for phototourism dataset')
    parser.add_argument('--use_cache', default=False, action="store_true",
                        help='whether to use ray cache (make sure img_downscale is the same)')

    # original NeRF parameters
    parser.add_argument('--N_emb_xyz', type=int, default=10,
                        help='number of xyz embedding frequencies')
    parser.add_argument('--N_emb_dir', type=int, default=4,
                        help='number of direction embedding frequencies')
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=64,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')

    # NeRF-W parameters
    parser.add_argument('--N_vocab', type=int, default=1500,
                        help='''number of vocabulary (number of images) 
                                in the dataset for nn.Embedding''')
    parser.add_argument('--encode_a', default=True, action="store_true",
                        help='whether to encode appearance (NeRF-A)')
    parser.add_argument('--N_a', type=int, default=48,
                        help='number of embeddings for appearance')
    parser.add_argument('--encode_t', default=True, action="store_true",
                        help='whether to encode transient object (NeRF-U)')
    parser.add_argument('--N_tau', type=int, default=16,
                        help='number of embeddings for transient objects')
    parser.add_argument('--beta_min', type=float, default=0.03,
                        help='minimum color variance for each ray')

    parser.add_argument('--chunk', type=int, default=32*1024*4,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')

    parser.add_argument('--video_format', type=str, default='gif',
                        choices=['gif', 'mp4'],
                        help='video format, gif or mp4')

    return parser.parse_args()


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, ts, N_samples, N_importance, use_disp,
                      chunk,
                      white_back=False,
                      a_emb=None,
                      **kwargs):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)

    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        ts[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        white_back,
                        test_time=True,
                        **kwargs)
        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = get_opts()
    kwargs = {'root_dir': args.root_dir,
              'split': args.split,
              'camera_noise': None, 
              }
    args.scene_name = args.ckpt_path.split('/')[-2]
    if args.dataset_name == 'blender':
        kwargs['img_wh'] = tuple(args.img_wh)
    else:
        kwargs['img_downscale'] = args.img_downscale
        # kwargs['use_cache'] = args.use_cache
    dataset = PhototourismDataset(**kwargs)
    scene = os.path.basename(args.root_dir.strip('/'))

    embedding_xyz = torch.nn.Identity()
    embedding_dir = torch.nn.Identity()
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}
    if args.encode_a:
        embedding_a = torch.nn.Embedding(args.N_vocab, args.N_a).cuda()
        load_ckpt(embedding_a, args.ckpt_path, model_name='embedding_a')
        embeddings['a'] = embedding_a
    if args.encode_t:
        embedding_t = torch.nn.Embedding(args.N_vocab, args.N_tau).cuda()
        load_ckpt(embedding_t, args.ckpt_path, model_name='embedding_t')
        embeddings['t'] = embedding_t

    nerf_coarse = NeRF('coarse',
                        xyz_L=args.N_emb_xyz,
                        dir_L=args.N_emb_dir).cuda()
    models = {'coarse': nerf_coarse}
    nerf_fine = NeRF('fine',
                     xyz_L=args.N_emb_xyz,
                     dir_L=args.N_emb_dir,
                     encode_appearance=args.encode_a,
                     in_channels_a=args.N_a,
                     encode_transient=args.encode_t,
                     in_channels_t=args.N_tau).cuda()



    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')

    models = {'coarse': nerf_coarse, 'fine': nerf_fine}

    imgs, depths, psnrs = [], [], []
    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    dataset.test_appearance_idx = 1123 # 85572957_6053497857.jpg
    dataset.test_appearance_idx = 1336 # 98007214_250229666.jpg  few30 - idx9
    kwargs = {}
    # define testing poses and appearance index for phototourism
    if args.dataset_name == 'phototourism' and args.split == 'test':
        # define testing camera intrinsics (hard-coded, feel free to change)
        dataset.test_img_w, dataset.test_img_h = args.img_wh
        dataset.test_focal = dataset.test_img_w/2/np.tan(np.pi/6) # fov=60 degrees
        dataset.test_K = np.array([[dataset.test_focal, 0, dataset.test_img_w/2],
                                   [0, dataset.test_focal, dataset.test_img_h/2],
                                   [0,                  0,                    1]])
        if scene == 'brandenburg_gate':
            # select appearance embedding, hard-coded for each scene
            # dataset.test_appearance_idx = 415 # 98007214_250229666.jpg  few30 - idx9
            
            # N_frames = 30*4
            N_frames = 30
            # dx = np.linspace(0, 0.03, N_frames)
            # dy = np.linspace(0, -0.1, N_frames)
            # dz = np.linspace(0, 0.5, N_frames)
            dx = np.linspace(-0.0, 0.6, N_frames)
            dy = np.linspace(0, 0, N_frames)
            dz = np.linspace(-0.0, -0.6, N_frames)
            # define poses
            dataset.poses_test = np.tile(dataset.poses_dict[1123], (N_frames, 1, 1))
            for i in range(N_frames):
                dataset.poses_test[i, 0, 3] += dx[i]
                dataset.poses_test[i, 1, 3] += dy[i]
                dataset.poses_test[i, 2, 3] += dz[i]
        else:
            raise NotImplementedError
        kwargs['output_transient'] = False

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample['rays']
        ts = sample['ts']
        results = batched_inference(models, embeddings, rays.cuda(), ts.cuda(),
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk,
                                    **kwargs)

        if args.dataset_name == 'blender':
            w, h = args.img_wh
        else:
            w, h = sample['img_wh']
        
        img_pred = np.clip(results['rgb_fine'].view(h, w, 3).cpu().numpy(), 0, 1)
        depth_pred_ = visualize_depth(results['depth_fine'].view(h, w), min_max=(0,5)).permute(1,2,0).numpy()
        
        img_pred_ = (img_pred*255).astype(np.uint8)
        imgs += [img_pred_]
        depth_pred = (depth_pred_*255).astype(np.uint8)
        depths += [depth_pred]
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}_depth.png'), depth_pred)

        if 'rgbs' in sample:
            rgbs = sample['rgbs']
            img_gt = rgbs.view(h, w, 3)
            # psnrs += [metrics.psnr(img_gt, img_pred).item()]
        
    if args.dataset_name == 'blender' or \
      (args.dataset_name == 'phototourism' and args.split == 'test'):
        imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}.{args.video_format}'),
                        # imgs, fps=30)
                        imgs, fps=10)
        imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}_depth.{args.video_format}'),
                        # imgs, fps=30)
                        depths, fps=10)
    
    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f'Mean PSNR : {mean_psnr:.2f}')