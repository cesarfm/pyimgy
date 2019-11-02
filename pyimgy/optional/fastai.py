from fastai.vision import Image, ImageSegment, image2np, pil2tensor

from pyimgy.optional.torch import *

DEFAULT_CONVERTER.type_converter_map.update({
    (Image, torch.Tensor): Image.data,
    (torch.Tensor, Image): Image,
    (Image, np.ndarray): lambda i: image2np(i.data),
    (np.ndarray, Image): lambda i: Image(pil2tensor(i, dtype=np.float32)),
    (np.ndarray, ImageSegment): lambda i: ImageSegment(pil2tensor(i, dtype=np.int64))
})
DEFAULT_CONVERTER.fallback_type_map.update({
    Image: torch.Tensor,
})
