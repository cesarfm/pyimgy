import torch

from pyimgy.core import *

# some monkey patches to the Tensor
torch.Tensor.transpose2 = torch.Tensor.transpose
torch.Tensor.copy = torch.Tensor.clone
torch.Tensor.tile = torch.Tensor.repeat


# monkey patches that overwrite existing methods have to be handled in a context...
class TorchPatchContext:
    def __enter__(self):
        torch.Tensor.transpose = torch.Tensor.permute
        torch.Tensor.repeat = torch.Tensor.repeat_interleave

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.Tensor.transpose = torch.Tensor.transpose2
        torch.Tensor.repeat = torch.Tensor.tile


DEFAULT_CONVERTER.type_converter_map.update({
    (np.ndarray, torch.Tensor): torch.from_numpy,
    (torch.Tensor, np.ndarray): lambda t: t.cpu().numpy(),
})
DEFAULT_CONVERTER.context_class = TorchPatchContext
