import PIL.Image

from pyimgy.core import *

__all__ = ['get_image_palette', 'show_image_palette', 'resize_as_pil', 'encode_array_channels', 'decode_array_channels', 'get_color_distribution']


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


@auto_plot(nrows=1, ncols=2)
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


def encode_array_channels(arr: np.ndarray, ch_dim: int = 2) -> np.ndarray:
    num_ch = arr.shape[ch_dim]
    assert arr.dtype == np.uint8, 'Only arrays of type uint8 can be encoded.'
    assert num_ch <= 4, 'Only a maximum of 4 channels can be encoded.'
    arr_enc = np.zeros(remove_at_index(arr.shape, ch_dim), dtype=np.uint32)
    for c in range(num_ch):
        arr_ch = np.take(arr, c, axis=ch_dim).astype(np.uint32)
        arr_enc = np.bitwise_or(arr_ch << (8 * c), arr_enc)
    return arr_enc


def decode_array_channels(arr_enc: np.ndarray, num_ch: int, ch_dim: int = -1) -> np.ndarray:
    assert arr_enc.dtype == np.uint32, 'Only arrays of type uint32 can be decoded.'
    assert num_ch <= 4, 'Only a maximum of 4 channels can be decoded.'
    arr_channels = [np.bitwise_and(arr_enc >> (8 * c), 255).astype(np.uint8) for c in range(num_ch)]
    return np.stack(arr_channels, axis=ch_dim)


def get_color_distribution(img, quantize_colors: Optional[int] = None, ratios: bool = False, sort: bool = True) -> Tuple[np.ndarray, ...]:
    if quantize_colors is None:
        img = convert_image(img, to_type=np.ndarray, shape='WHC')
        total_pixels = img.shape[0] * img.shape[1]
        num_ch = img.shape[2]
        if num_ch == 1:
            unique_colors, color_count = np.unique(img, return_counts=True)
        else:
            # we will do a trick and put all the channels in the same long integer
            enc_img = encode_array_channels(img, ch_dim=2)
            unique_enc_colors, color_count = np.unique(enc_img, return_counts=True)
            # now we have to "decode" the flattened channels
            unique_colors = decode_array_channels(unique_enc_colors, num_ch=num_ch)
    else:
        pal, pal_img = get_image_palette(img, quantize_colors)
        img = convert_image(pal_img, to_type=np.ndarray)
        total_pixels = img.shape[0] * img.shape[1]
        unique_pal_colors, color_count = np.unique(img, return_counts=True)
        unique_colors = pal[unique_pal_colors]

    if sort:
        # we want to sort these values by descending count
        sort_idxs = np.argsort(-color_count)
        unique_colors = unique_colors[sort_idxs]
        color_count = color_count[sort_idxs]

    return unique_colors, (color_count / total_pixels if ratios else color_count)


# RESIZING

# this method will keep the type of the input, only making conversions when needed
@converting(to=PILImage, preserve_type=True)
def resize_as_pil(img: PILImage, width: int, height: int, resample: int = 0):
    return img.resize((width, height), resample)
