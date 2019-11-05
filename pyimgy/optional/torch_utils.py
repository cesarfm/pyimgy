import torch
import torch.nn.functional as F

from pyimgy.core import *


@converting(shape='11WH', preserve_shape=True)
def resize_as_torch(img: torch.Tensor, width: int, height: int, mode: str):
    # the interpolation needs a 4-D tensor (B, C, H, W), it should be brought back to the original shape
    return F.interpolate(img, size=(height, width), mode=mode, align_corners=None if mode in ('nearest', 'area') else False)
