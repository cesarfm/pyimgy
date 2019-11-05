import torch

from pyimgy.core import *

# some monkey patches to the Tensor
torch.Tensor.transpose2 = torch.Tensor.transpose
torch.Tensor.transpose = torch.Tensor.permute
torch.Tensor.copy = torch.Tensor.clone
torch.Tensor.tile = torch.Tensor.repeat
torch.Tensor.repeat = torch.Tensor.repeat_interleave

DEFAULT_CONVERTER.type_converter_map.update({
    (np.ndarray, torch.Tensor): torch.from_numpy,
    (torch.Tensor, np.ndarray): lambda t: t.cpu().numpy(),
})
