import torch
from kornia.losses import ssim as dssim
from barf_vis import *
import camera

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def ssim(image_pred, image_gt, reduction='mean'):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    dssim_ = dssim(image_pred, image_gt, 3, reduction) # dissimilarity in [0, 1]
    return 1-2*dssim_ # in [-1, 1]

## pose
def parse_raw_camera(pose_raw):
    pose_flip = camera.pose(R=torch.diag(torch.tensor([1,-1,-1])))
    pose = camera.pose.compose([pose_flip,pose_raw[:3]])
    pose = camera.pose.invert(pose)
    pose = camera.pose.compose([pose_flip,pose])
    return pose

def prealign_cameras(pose,pose_GT, scaling=True):
    pose, pose_GT = pose.float(), pose_GT.float()
    center = torch.zeros(1,1,3)
    center_pred = camera.cam2world(center,pose)[:,0] # [N,3]
    center_GT = camera.cam2world(center,pose_GT)[:,0] # [N,3]
    sim3 = camera.procrustes_analysis(center_GT,center_pred)
    if scaling is True:
        center_aligned = (center_pred-sim3.t1)/sim3.s1@sim3.R.t()*sim3.s0 + sim3.t0
    else:
        center_aligned = (center_pred-sim3.t1)@sim3.R.t() + sim3.t0
    R_aligned = pose[...,:3]@sim3.R.t()
    t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
    aligned_pose = camera.pose(R=R_aligned,t=t_aligned)
    return aligned_pose,sim3

def evaluate_camera_alignment(pose_aligned,pose_GT):
    # measure errors in rotation and translation
    R_aligned,t_aligned = pose_aligned.split([3,1],dim=-1)
    R_GT,t_GT = pose_GT.split([3,1],dim=-1)
    R_error = camera.rotation_distance(R_aligned,R_GT)
    t_error = (t_aligned-t_GT)[...,0].norm(dim=-1)
    error = dict(R=R_error,t=t_error)
    return error

def pose_metric(refine_poses, gt_poses):
    refine_poses = torch.stack([parse_raw_camera(p) for p in refine_poses.float()],dim=0)
    gt_poses = torch.stack([parse_raw_camera(p) for p in gt_poses.float()],dim=0)
    aligned_pose, sim3 = prealign_cameras(refine_poses, gt_poses)
    error = evaluate_camera_alignment(aligned_pose, gt_poses)
    return error

def evaluate_camera_alignment(pose_aligned,pose_GT):
    # measure errors in rotation and translation
    R_aligned,t_aligned = pose_aligned.split([3,1],dim=-1)
    R_GT,t_GT = pose_GT.split([3,1],dim=-1)
    R_error = camera.rotation_distance(R_aligned,R_GT)
    t_error = (t_aligned-t_GT)[...,0].norm(dim=-1)
    error = dict(R=R_error,t=t_error)
    return error

def parse_raw_camera(pose_raw):
    pose_flip = camera.pose(R=torch.diag(torch.tensor([1,-1,-1])))
    pose = camera.pose.compose([pose_flip,pose_raw[:3]])
    pose = camera.pose.invert(pose)
    return pose

def generate_videos_pose(path, pose=None, pose_ref=None, sample_nums=None, cam_depth=0.5, scaling=True, connect=False):
    assert not (pose is None and pose_ref is None)
    if sample_nums is None:
        sample_nums = len(pose) if pose is not None else len(pose_ref)
        if pose is not None:
            pose = pose[:sample_nums]
        if pose_ref is not None:
            pose_ref = pose_ref[:sample_nums]
    fig = plt.figure(figsize=(10,10))
    if pose is not None and pose_ref is not None:
        pose_aligned, _ = prealign_cameras(pose, pose_ref, scaling=scaling)
    else:
        pose_aligned = pose
    if pose_aligned is not None:
        pose_aligned = pose_aligned.detach().cpu()
    if pose_ref is not None:
        pose_ref = pose_ref.detach().cpu()
    plot_save_poses(fig, path, pose=pose_aligned, pose_ref=pose_ref, cam_depth=cam_depth, connect=connect)
    plt.close()

def plot_save_poses(fig, path, pose=None, pose_ref=None, cam_depth=0.5, connect=False):
    # get the camera meshes
    assert not (pose is None and pose_ref is None)
    if pose is not None:
        _,_,cam = get_camera_mesh(pose,depth=cam_depth)
        cam = cam.numpy()
    if pose_ref is not None:
        _,_,cam_ref = get_camera_mesh(pose_ref,depth=cam_depth)
        cam_ref = cam_ref.numpy()
    # set up plot window(s)
    ax = fig.add_subplot(111,projection="3d")
    setup_3D_plot(ax,elev=45,azim=35,lim=edict(x=(-3,3),y=(-3,3),z=(-3,2.4)))
    plt.subplots_adjust(left=0,right=1,bottom=0,top=0.95,wspace=0,hspace=0)
    plt.margins(tight=True,x=0,y=0)
    # plot the cameras
    N = len(cam) if pose is not None else len(cam_ref)
    ref_color = (0.7,0.2,0.7)
    pred_color = (0,0.6,0.7)
    if pose_ref is not None:
        ax.add_collection3d(Poly3DCollection([v[:4] for v in cam_ref],alpha=0.2,facecolor=ref_color))
        for i in range(N):
            ax.plot(cam_ref[i,:,0],cam_ref[i,:,1],cam_ref[i,:,2],color=ref_color,linewidth=0.5)
            ax.scatter(cam_ref[i,5,0],cam_ref[i,5,1],cam_ref[i,5,2],color=ref_color,s=20)
    if pose is not None:
        ax.add_collection3d(Poly3DCollection([v[:4] for v in cam],alpha=0.2,facecolor=pred_color))
        for i in range(N):
            ax.plot(cam[i,:,0],cam[i,:,1],cam[i,:,2],color=pred_color,linewidth=1)
            ax.scatter(cam[i,5,0],cam[i,5,1],cam[i,5,2],color=pred_color,s=20)
    if connect is True:
        assert not (pose is None or pose_ref is None)
        for i in range(N):
            ax.plot([cam[i,5,0],cam_ref[i,5,0]],
                    [cam[i,5,1],cam_ref[i,5,1]],
                    [cam[i,5,2],cam_ref[i,5,2]],color=(1,0,0),linewidth=3)
    plt.savefig(path, dpi=75)
    plt.clf()