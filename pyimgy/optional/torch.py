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


def normalize_tensor_to_ints(arr: torch.Tensor) -> torch.Tensor:
    is_norm1 = arr.dtype in (torch.float32, torch.float64) and arr.max() <= 1
    return (arr * 255 if is_norm1 else arr).type(torch.uint8)


def normalize_tensor_to_floats(arr: torch.Tensor) -> torch.Tensor:
    arr = arr.type(torch.float32)
    return arr / 255 if arr.max() > 1 else arr


DEFAULT_CONVERTER.type_converter_map.update({
    (np.ndarray, torch.Tensor): torch.from_numpy,
    (torch.Tensor, np.ndarray): lambda t: t.cpu().detach().numpy(),
})
DEFAULT_CONVERTER.norm_map.update({
    ('int_255', torch.Tensor): normalize_tensor_to_ints,
    ('float_1', torch.Tensor): normalize_tensor_to_floats
})
DEFAULT_CONVERTER.context_class = TorchPatchContext
