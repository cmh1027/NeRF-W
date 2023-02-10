import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

def visualize_depth(depth, cmap=cv2.COLORMAP_JET, min_max=None):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    if min_max is None:
        mi = np.min(x) # get minimum depth
        ma = np.max(x)
    else:
        mi, ma = min_max
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_