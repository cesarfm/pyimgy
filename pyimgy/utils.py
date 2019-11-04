from typing import Tuple

import PIL.Image

from pyimgy.core import *

__all__ = ['is_valid_image_shape', 'assert_valid_image_shape', 'get_image_palette', 'show_image_palette']


# IMAGE FORMAT UTILS


def is_valid_image_shape(img: np.ndarray) -> bool:
    return img.ndim == 2 or img.ndim == 3 and img.shape[2] in (1, 3, 4)


def assert_valid_image_shape(img: np.ndarray) -> None:
    assert is_valid_image_shape(img), f'Invalid image shape: {img.shape}'


# COLORS, PALETTE

@converting(to=PILImage)
def get_image_palette(img: PILImage, colors: int = 256) -> Tuple[np.ndarray, PILImage]:
    """
    For the given image, return a palette of the request size, with the optimal colors. It leverages Pillow's palette conversion.

    :param img: an image
    :param colors: palette size, between 1 to 256
    :return: a tuple of
     - a numpy array of shape (C x 3), where C is the palette size
     - the image with the reduced palette, as a PIL Image
    """
    assert 1 <= colors <= 256
    pal_img = img.convert('P', colors=colors, palette=PIL.Image.ADAPTIVE)
    # get the palette and give it the right shape
    pal = np.array(pal_img.getpalette()[:colors * 3]).reshape((-1, 3))
    return pal, pal_img


@auto_axes(nrows=1, ncols=2)
def show_image_palette(img, colors: int = 256, ax=None) -> None:
    pal, pal_img = get_image_palette(img, colors)
    # enlarge it to a reasonable size
    width, height = pal_img.size
    if width // colors > 1:
        pal = np.repeat(pal, width // colors, axis=0)
    pal = np.broadcast_to(pal, (height, pal.shape[0], 3))

    for a in ax: a.axis('off')
    ax[0].imshow(pal_img)
    ax[0].set_title(f'Image {pal_img.size}')
    ax[1].imshow(pal)
    ax[1].set_title(f'Palette, {colors} colors')
