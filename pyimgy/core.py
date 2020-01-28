from functools import wraps
from typing import Union, Tuple, Any, Optional

import PIL.Image
import numpy as np
from PIL.Image import Image as PILImage


def replace_at_index(tup: tuple, idx: int, value) -> tuple:
    assert 0 <= idx < len(tup)
    return tup[:idx] + (value,) + tup[idx + 1:]


def remove_at_index(tup: tuple, idx: int) -> tuple:
    assert 0 <= idx < len(tup)
    return tup[:idx] + tup[idx + 1:]


def get_item_for_type_of(obj, type_map, default=(None, None)):
    return next((t for t in type_map.items() if isinstance(obj, t[0])), default)


def make_nested_map(mapping):
    nested = {}
    for keys, value in mapping.items():
        row = nested
        for k in keys[:-1]:
            row = row.setdefault(k, {})
        row[keys[-1]] = value
    return nested


def get_sub_map(mapping, key, key_idx):
    return {remove_at_index(keys, key_idx): value for keys, value in mapping.items() if keys[key_idx] == key}


# CONVERSION UTILS


class _DummyPatchContext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class ImageConverter:
    VALID_DIMENSIONS = (2, 3, 4)
    VALID_CHANNELS = (1, 3, 4)
    SHAPE_PARAMS_MAP = {
        'WH': (2, 1, False),
        '3WH': (3, 3, False),
        '1WH': (3, 1, False),
        'WH3': (3, 3, True),
        'WH1': (3, 1, True),
        'WHC': (3, None, True),
        'CWH': (3, None, False),
        '13WH': (4, 3, False),
        '1WH3': (4, 3, True),
        '11WH': (4, 1, False),
        '1CWH': (4, None, False)
    }
    SHAPE_PARAMS_INVERSE_MAP = {v: k for k, v in SHAPE_PARAMS_MAP.items()}

    def __init__(self, type_converter_map, norm_map, fallback_type_map, context_class=_DummyPatchContext):
        self.type_converter_map = type_converter_map  # (from_type, to_type) -> type conversion func
        self.norm_map = norm_map  # (norm) -> (type) -> normalizer func
        self.fallback_type_map = fallback_type_map
        self.context_class = context_class

        self._converter_cache = {}

    def _get_type_converter(self, from_, to_type):
        if from_ is None or to_type is None:
            return None
        from_is_type = type(from_) == type
        from_type = from_ if from_is_type else type(from_)
        converter_func = self.type_converter_map.get((from_type, to_type))
        if converter_func is None:
            converter_func = self._converter_cache.get((from_type, to_type))
        if converter_func is None and not from_is_type:
            converter_func = next((conv for (ft, tt), conv in self.type_converter_map.items() if tt == to_type and isinstance(from_, ft)), None)
            if converter_func is not None:
                self._converter_cache[(from_type, to_type)] = converter_func
        return converter_func

    def _get_fallback_type_of_obj(self, obj):
        return get_item_for_type_of(obj, self.fallback_type_map)[1]

    def _convert_type(self, img, to_type, chained=False):
        if isinstance(img, to_type):
            return img

        from_type = type(img)
        converter_func = self._get_type_converter(img, to_type)

        if converter_func is not None:
            return converter_func(img)
        elif chained:
            from_fb_type = self.fallback_type_map.get(from_type)
            to_fb_type = self.fallback_type_map.get(to_type)
            if self._get_type_converter(from_fb_type, to_type) is not None:
                return self._convert_type_chained(img, (from_fb_type, to_type))
            elif self._get_type_converter(from_type, to_fb_type) is not None:
                return self._convert_type_chained(img, (to_fb_type, to_type))
            elif self._get_type_converter(from_fb_type, to_fb_type) is not None:
                return self._convert_type_chained(img, (from_fb_type, to_fb_type, to_type))
        else:
            raise Exception(f'Not possible to convert from {from_type} to {to_type}!!!')

    def _convert_type_chained(self, img, to_types):
        from_ = img
        for to_type in to_types:
            converter = self._get_type_converter(from_, to_type)
            img = converter(img)
            from_ = to_type
        return img

    def _get_array_shape(self, arr) -> Tuple[str, int, bool]:
        assert arr.ndim in (2, 3) or arr.ndim == 4 and arr.shape[0] == 1, f'Invalid shape: {arr.shape}'
        if arr.ndim == 2:
            num_ch = 1
            trailing_ch = False
        else:
            leading_ch = arr.shape[-3] in self.VALID_CHANNELS
            trailing_ch = arr.shape[-1] in self.VALID_CHANNELS
            assert leading_ch ^ trailing_ch, f'Not possible to infer channel from shape: {arr.shape}'
            num_ch = arr.shape[-1 if trailing_ch else -3]
        shape_args = (arr.ndim, num_ch if num_ch in (1, 3) else None, trailing_ch)
        return self.SHAPE_PARAMS_INVERSE_MAP.get(shape_args), num_ch, trailing_ch

    def get_array_shape(self, arr) -> str:
        return self._get_array_shape(arr)[0]

    def _format_array_shape(self, arr, out_ndim: int = 3, out_num_ch: int = None, out_trailing_ch: bool = False):
        assert out_ndim in self.VALID_DIMENSIONS, f'Invalid target dimensions: {out_ndim}'
        assert out_num_ch is None or out_num_ch in self.VALID_CHANNELS, f'Invalid target channel: {out_num_ch}'

        old_shape = arr.shape
        _, num_ch, trailing_ch = self._get_array_shape(arr)

        if arr.ndim == 4:
            arr = arr.squeeze(0)
        elif arr.ndim == 2:
            arr = arr[None]  # unsqueeze(0)

        # at this point, ndim can be only 3
        ch_dim = 2 if trailing_ch else 0

        if out_ndim == 2:
            assert num_ch == 1, f'Not possible to make 2-D with {num_ch} channels, for shape {old_shape}'
            return arr.squeeze(ch_dim)

        # now make sure that the channels are correct
        if out_num_ch is not None:
            if num_ch == 1 and out_num_ch > 1:
                arr = arr.repeat(out_num_ch, ch_dim)
            else:
                assert num_ch == out_num_ch, f'Not possible to change channels from {num_ch} to {out_num_ch} for shape {old_shape}'

        # transpose if needed
        if ch_dim == 0 and out_trailing_ch:
            arr = arr.transpose((1, 2, 0))
        if ch_dim == 2 and not out_trailing_ch:
            arr = arr.transpose((2, 0, 1))

        return arr[None] if out_ndim == 4 else arr

    def _convert_shape(self, img, shape: str) -> np.ndarray:
        shaper_args = self.SHAPE_PARAMS_MAP.get(shape)
        assert shaper_args is not None, f'Unrecognized shape: {shape}'
        shaper_type = self._get_fallback_type_of_obj(img)
        if shaper_type is not None:
            img = self._convert_type(img, to_type=shaper_type)
        return self._format_array_shape(img, *shaper_args)

    def _convert_norm(self, img, norm):
        norm_type_map = get_sub_map(self.norm_map, norm, 0)
        assert len(norm_type_map) > 0, f'Unrecognized normalization: {norm} for type {type(img)}'
        norm_type, norm_func = get_item_for_type_of(img, norm_type_map)
        if norm_func is None:
            norm_type = self._get_fallback_type_of_obj(img)
            norm_func = norm_type_map.get(norm_type)
        assert norm_func is not None, f'No available normalization for type {type(img)} and norm {norm}'
        img = self._convert_type(img, to_type=norm_type)
        return norm_func(img)

    def convert_image(self, img, to_type=None, shape=None, norm=None):
        with self.context_class():
            original_type = type(img)
            if shape:
                img = self._convert_shape(img, shape)
            if norm:
                img = self._convert_norm(img, norm)
            return self._convert_type(img, to_type=to_type or original_type, chained=True)


def normalize_numpy_to_ints(arr: np.ndarray) -> np.ndarray:
    is_norm1 = arr.dtype in (np.float32, np.float64) and arr.max() <= 1
    return (arr * 255 if is_norm1 else arr).astype(np.uint8)


def normalize_numpy_to_floats(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    return arr / 255 if arr.max() > 1 else arr


DEFAULT_CONVERTER = ImageConverter(
    type_converter_map={
        (np.ndarray, PILImage): PIL.Image.fromarray,
        (PILImage, np.ndarray): np.array
    },
    norm_map={
        ('int_255', np.ndarray): normalize_numpy_to_ints,
        ('float_1', np.ndarray): normalize_numpy_to_floats
    },
    fallback_type_map={
        PILImage: np.ndarray
    }
)


def convert_image(img: Any, to_type: Optional[type] = None, shape: Optional[str] = None, norm: Optional[str] = None) -> Any:
    """
    This function allows to convert a given image data (which could correspond to one image or a batch), from various Python types, shapes and value
    normalization. Basic usage:

    ```
    from pyimgy.core import *
    img = PIL.Image.open('images/bedroom-with-border.jpg')
    converted_img = convert_image(img, to_type=np.ndarray, shape='3WH', norm='int_255')
    ```

    The arguments `to_type`, `shape` and `norm` are all optional, but at least one must be given.

    :param img:
    :param to_type: indicates the Python type to which to convert the image (i.e. the type of the output). Possible values so far (from and to):
                    <br/> - `numpy.ndarray`
                    <br/> - `PIL.Image.Image` (the Image class from Pillow)
                    <br/> - `torch.Tensor` (optional, if PyTorch is installed)
                    <br/> - `fast.vision.Image` (optional, if fast.ai is installed)
    :param shape: is used to re-shape the image data to one of the standard representations, given by a key. The number of character in the key tells
                  how many dimensions the output shape should have, and the characters themselves determine what each dimension is.
                  <br/> - 2 dim: `WH` (width, height)
                  <br/> - 3 dim: `1WH, 3WH, CWH, WH1, WH3, WHC` (the channel can have size 1 or 3, or `C` as a wildcard, can be leading or trailing)
                  <br/> - 4 dim: `13WH, 11WH, 1CWH, 1WH3` (the first dimension is always the batch size)
    :param norm: to specify a normalization for the data itself
                 <br/> - `int_255`: integer, scaled so the max value is 255
                 <br/> - `float_1`: float, scaled so the max value is 1.0
    :return:
    """
    return DEFAULT_CONVERTER.convert_image(img, to_type, shape, norm)


def get_array_shape(img) -> str:
    return DEFAULT_CONVERTER.get_array_shape(img)


def convert_to_standard_pil(img):
    return img if isinstance(img, PILImage) else convert_image(img, to_type=PILImage, shape='WHC', norm='int_255')


def convert_for_plot(img):
    return img if isinstance(img, PILImage) else convert_image(img, to_type=np.ndarray, shape='WH3')


def converting(to: type = None, shape: str = None, norm: str = None, argument: Union[int, str] = 0, return_pos: int = 0, preserve_type: bool = False,
               preserve_shape: bool = False):
    """
    A neat decorator to simplify common conversions in functions.

    :param to: The type to which the indicated argument should be converted BEFORE calling the function.
    :param shape:
    :param norm:
    :param argument: The position of the argument to convert. If it's an integer, it is taken from the positional args; if it's a string from the
                     named kwargs.
    :param return_pos: If preserve_type=True, the position of the corresponding output that has to be converted back.
    :param preserve_type: if True, the type of the output is preserved, i.e. it is converted back to the original type of the input, if needed.
    :param preserve_shape: if True, the shape of the output is preserved from the input; if it's a recognized shape.
    :return:
    """

    def convert_decorator(func):
        @wraps(func)
        def decorated(*args, **kwargs):
            input = args[argument] if isinstance(argument, int) else kwargs[argument]
            output_type = type(input) if preserve_type else None
            output_shape = DEFAULT_CONVERTER.get_array_shape(input) if preserve_shape else None
            input_converted = DEFAULT_CONVERTER.convert_image(input, to_type=to, shape=shape, norm=norm)

            if isinstance(argument, int):
                # args is a tuple, so we cannot modify it directly
                args = replace_at_index(args, argument, input_converted)
            else:
                kwargs[argument] = input_converted

            outputs = func(*args, **kwargs)

            if not (preserve_type or preserve_shape):
                return outputs
            elif isinstance(outputs, tuple):
                return replace_at_index(outputs, return_pos, convert_image(outputs[return_pos], to_type=output_type, shape=output_shape))
            else:
                return convert_image(outputs, to_type=output_type, shape=output_shape)

        return decorated

    return convert_decorator


# PLOTTING UTILS

def auto_plot(name: str = 'ax', **plot_kwargs):
    assert name in ('ax', 'fig'), f'Invalid name: {name}'

    def auto_axes_decorator(func):
        @wraps(func)
        def decorated(*args, **kwargs):
            _plt = None
            if kwargs.get(name) is None:
                import matplotlib.pyplot as _plt
                if name == 'ax':
                    if len(plot_kwargs) == 0:
                        kwargs[name] = _plt.axes()
                    else:
                        _, ax = _plt.subplots(**plot_kwargs)
                        kwargs[name] = ax
                elif name == 'fig':
                    kwargs[name] = _plt.figure(**plot_kwargs)

            out = func(*args, **kwargs)

            if _plt:
                _plt.show()
            return out

        return decorated

    return auto_axes_decorator


@auto_plot()
def show_image(img, title=None, ax=None):
    ax.imshow(convert_for_plot(img))
    ax.axis('off')
    if title is not None:
        ax.set_title(str(title))


@auto_plot(name='fig', figsize=(15, 8))
def show_images(imgs, titles=None, r=1, fig=None):
    if isinstance(titles, list):
        assert len(imgs) == len(titles)
    else:
        titles = [titles] * len(imgs)

    c = np.ceil(len(imgs) / r)
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(r, c, i + 1)
        show_image(img, titles[i], ax=ax)
