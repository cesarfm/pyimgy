import torch

from pyimgy.core import *


class TensorArray(BaseArray):
    raw_type = torch.Tensor
    delegated_methods = {
        'broadcast_to': torch.Tensor.expand,
        'transpose': torch.Tensor.permute,
        'copy': torch.Tensor.clone
    }
    delegated_properties = {
        'ndim': torch.Tensor.ndimension
    }

    def __init__(self, value):
        super(TensorArray, self).__init__(value)
