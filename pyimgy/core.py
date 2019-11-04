from functools import wraps
from typing import Union

import PIL.Image
import numpy as np
from PIL.Image import Image as PILImage

# CONVERSION UTILS

IMAGE_CONVERSION_MAP = {
    (np.ndarray, PILImage): PIL.Image.fromarray,
    (PILImage, np.ndarray): np.array
}


def convert_image(img, to_type):
    if isinstance(img, to_type):
        return img
    from_type = type(img)
    converter = IMAGE_CONVERSION_MAP.get((from_type, to_type))
    if converter is None:
        raise Exception(f'Not possible to convert from {from_type} to {to_type}!!!')
    return converter(img)


def replace_at_index(tup: tuple, idx: int, value) -> tuple:
    assert 0 <= idx < len(tup)
    return tup[:idx] + (value,) + tup[idx + 1:]


def converting(to: type, argument: Union[int, str] = 0, return_pos: int = 0, preserve_type: bool = False):
    """
    A neat decorator to simplify common conversions in functions.

    :param to: The type to which the indicated argument should be converted BEFORE calling the function.
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
            input_converted = convert_image(input, to_type=to)

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
