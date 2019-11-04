import types
from functools import wraps
from typing import Union

import PIL.Image
import numpy as np
from PIL.Image import Image as PILImage

# CONVERSION UTILS

VALID_CHANNELS = (1, 3)


class BaseArray:
    raw_type = np.ndarray
    delegated_methods = {
        'broadcast_to': np.broadcast_to,
        'unsqueeze': np.expand_dims
    }
    delegated_properties = {}

    def __init__(self, value):
        self.value = value

    @classmethod
    def wrap(cls, value):
        return cls(value) if isinstance(value, cls.raw_type) else value

    def __getattr__(self, item):
        attr = self.delegated_properties.get(item)
        if attr is not None:
            return self.wrap(attr(self))
        attr = self.delegated_methods.get(item) or self.value.__getattribute__(item)
        if isinstance(attr, (types.FunctionType, types.BuiltinFunctionType, types.MethodType, types.BuiltinMethodType)):
            # callable -- needs a wrapper
            def _wrapper(*args, **kwargs):
                return self.wrap(attr(*args, **kwargs))

            return _wrapper
        return self.wrap(attr)


def format_shape_universal(arr, out_ch: int = 3, trailing_ch: bool = False, include_bn: bool = False):
    old_shape = arr.shape
    assert out_ch in VALID_CHANNELS, f'Invalid target channel: {out_ch}'
    assert arr.ndim in (2, 3) or arr.ndim == 4 and old_shape[0] == 1, f'Invalid shape: {old_shape}'

    if arr.ndim == 4:
        arr = arr.squeeze(0)

    # at this point, ndim can be only 2 or 3
    hw = old_shape[-2:]
    if arr.ndim == 3:
        d1_like_ch = arr.shape[0] in VALID_CHANNELS
        d3_like_ch = arr.shape[2] in VALID_CHANNELS
        assert d1_like_ch ^ d3_like_ch, f'Ambiguous shape: {old_shape}'

        if d1_like_ch and trailing_ch:
            arr = arr.transpose((1, 2, 0))
        if d3_like_ch and not trailing_ch:
            arr = arr.transpose((2, 0, 1))
        if d3_like_ch:
            hw = old_shape[-3:-1]

    new_shape = (hw + (out_ch,)) if trailing_ch else ((out_ch,) + hw)
    if arr.shape != new_shape:
        arr = arr.broadcast_to(new_shape).copy()
    if include_bn:
        arr = arr.unsqueeze(0)
    return arr


def normalize_numpy_to_ints(arr: np.ndarray) -> np.ndarray:
    is_norm1 = arr.dtype in (np.float32, np.float64) and arr.max() <= 1
    return (arr * 255 if is_norm1 else arr).astype(np.uint8)


def normalize_numpy_to_floats(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    return arr / 255 if arr.max() > 1 else arr


class ImageConverter:
    _shaper_params_map = {
        '3WH': (3, False, False),
        '1WH': (1, False, False),
        'WH3': (3, True, False),
        'WH1': (1, True, False),
        '13WH': (3, False, True),
        '1WH3': (3, True, True)
    }

    def __init__(self, formats_map, shapers_map, norm_map, default_type):
        self.formats_map = formats_map  # (from_type, to_type) -> type conversion func
        self.shapers_map = shapers_map  # (type) -> shaper func
        self.norm_map = norm_map  # (type, norm) -> normalizer func
        self.default_type = default_type

    def _convert_format(self, img, to_type):
        if isinstance(img, to_type):
            return img
        from_type = type(img)
        converter_func = self.formats_map.get((from_type, to_type))
        assert converter_func is not None, f'Not possible to convert from {from_type} to {to_type}!!!'
        return converter_func(img)

    def _convert_shape(self, img, shape: str) -> np.ndarray:
        shaper_args = self._shaper_params_map.get(shape)
        assert shaper_args is not None, f'Unrecognized shape: {shape}'
        shaper_type = next((t for t in self.shapers_map.keys() if isinstance(img, t)), self.default_type)
        img = self._convert_format(img, to_type=shaper_type)
        return self.shapers_map[shaper_type](img, *shaper_args)

    def _convert_norm(self, img, norm):
        norm_type = next((t for t, n in self.norm_map.keys() if n == norm and isinstance(img, t)), self.default_type)
        norm_func = self.norm_map.get((norm_type, norm))
        assert norm_func is not None, f'Not possible to normalize for norm {norm} and type {norm_type}'
        img = self._convert_format(img, to_type=norm_type)
        return norm_func(img)

    def convert_image(self, img, to_type=None, shape=None, norm=None):
        original_type = type(img)
        if shape:
            img = self._convert_shape(img, shape)
        if norm:
            img = self._convert_norm(img, norm)
        return self._convert_format(img, to_type=to_type or original_type)


DEFAULT_CONVERTER = ImageConverter(
    formats_map={
        (np.ndarray, PILImage): PIL.Image.fromarray,
        (PILImage, np.ndarray): np.array
    },
    shapers_map={
        np.ndarray: format_shape_universal
    },
    norm_map={
        (np.ndarray, 'int_255'): normalize_numpy_to_ints,
        (np.ndarray, 'float_1'): normalize_numpy_to_floats
    },
    default_type=np.ndarray
)


def convert_image(*args, **kwargs):
    return DEFAULT_CONVERTER.convert_image(*args, **kwargs)


def replace_at_index(tup: tuple, idx: int, value) -> tuple:
    assert 0 <= idx < len(tup)
    return tup[:idx] + (value,) + tup[idx + 1:]


def converting(to: type, shape=None, norm=None, argument: Union[int, str] = 0, return_pos: int = 0, preserve_type: bool = False):
    """
    A neat decorator to simplify common conversions in functions.

    :param to: The type to which the indicated argument should be converted BEFORE calling the function.
    :param shape:
    :param norm:
    :param argument: The position of the argument to convert. If it's an integer, it is taken from the positional args; if it's a string from the
                     named kwargs.
    :param return_pos: If preserve_type=True, the position of the corresponding output that has to be converted back.
    :param preserve_type: if True, the type of the output is preserved, i.e. it is converted back to the original type of the input, if needed.
    :return:
    """

    def convert_decorator(func):
        @wraps(func)
        def decorated(*args, **kwargs):
            input = args[argument] if isinstance(argument, int) else kwargs[argument]
            original_type = type(input)
            input_converted = convert_image(input, to_type=to, shape=shape, norm=norm)

            if isinstance(argument, int):
                # args is a tuple, so we cannot modify it directly
                args = replace_at_index(args, argument, input_converted)
            else:
                kwargs[argument] = input_converted

            outputs = func(*args, **kwargs)

            if not preserve_type:
                return outputs
            elif isinstance(outputs, tuple):
                return replace_at_index(outputs, return_pos, convert_image(outputs[return_pos], to_type=original_type))
            else:
                return convert_image(outputs, to_type=original_type)

        return decorated

    return convert_decorator


# PLOTTING UTILS

def auto_axes(name: str = 'ax', **subplot_kwargs):
    def auto_axes_decorator(func):
        @wraps(func)
        def decorated(*args, **kwargs):
            _plt = None
            if kwargs.get(name) is None:
                import matplotlib.pyplot as _plt
                if len(subplot_kwargs) == 0:
                    kwargs[name] = _plt.axes()
                else:
                    _, ax = _plt.subplots(**subplot_kwargs)
                    kwargs[name] = ax

            out = func(*args, **kwargs)

            if _plt:
                _plt.show()
            return out

        return decorated

    return auto_axes_decorator
