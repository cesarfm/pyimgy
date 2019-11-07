import PIL.Image

from pyimgy.core import *

__all__ = ['ImageCropper', 'crop_image', 'crop_image_from_file']


class ImageCropper:
    """
    Component to automatically remove border frames of the same color from an image.
    """

    def __init__(self, img: Union[np.ndarray, PILImage], quantize_colors: Optional[int] = 256, tolerance: float = None):
        """
        Creates the cropper for a given image.

        :param img: image to crop
        :param quantize_colors: if not None, the number of colors to reduce the palette to; it helps to remove noise (int)
        :param tolerance: how far apart two channels/colors can be to be considered the same (float)
        """
        self.original_img = img
        self.tolerance = tolerance or ((quantize_colors or 256) // 16)

        if quantize_colors:
            pil_img = convert_image(img, to_type=PILImage)
            img = pil_img.convert('P', palette=PIL.Image.ADAPTIVE, colors=quantize_colors)

        self.img = convert_image(img, to_type=np.ndarray, shape='WHC')
        self.img_t = self.img.transpose((1, 0, 2))

    def get_unique_value(self, arr: np.ndarray):
        a_min, a_max = arr.min(), arr.max()
        if a_min == a_max:
            return a_min
        elif a_max - a_min <= self.tolerance:
            return np.median(arr)
        else:
            return None

    def get_unique_color(self, arr: np.ndarray):
        if arr.ndim <= 2 or arr.shape[2] == 1:
            return self.get_unique_value(arr)
        else:
            num_channels = arr.shape[2]
            color = np.zeros(num_channels)
            for c in range(num_channels):
                channel = self.get_unique_value(arr[:, :, c])
                if channel is None: return None
                color[c] = channel
            return color

    def are_colors_equal(self, c1, c2) -> bool:
        if c1 is None or c2 is None:
            return False
        elif self.tolerance == 0:
            return np.all(c1 == c2)
        else:
            return np.sum((c1 - c2) ** 2) <= self.tolerance ** 2

    def get_crop_position(self, previous_ref_color=None, vertical: bool = False, from_end: bool = False, until_pos: int = None):
        img = self.img_t if vertical else self.img
        pos = img.shape[0] - 1 if from_end else 0
        until_pos = until_pos or (img.shape[0] - 1 - pos)
        step = -1 if from_end else 1

        # get color of first line -- this will be the reference
        ref_color = self.get_unique_color(img[None, pos, :])  # the None is needed for keeping the dims

        if ref_color is None or not (previous_ref_color is None or self.are_colors_equal(ref_color, previous_ref_color)):
            return pos + (1 if from_end else 0), None

        while pos != until_pos and self.are_colors_equal(ref_color, self.get_unique_color(img[None, pos + step, :])):
            pos += step

        return pos + (0 if from_end else 1), ref_color

    def get_cropping_box(self):
        top, ref_color = self.get_crop_position()
        left, ref_color = self.get_crop_position(ref_color, vertical=True)
        bottom, ref_color = self.get_crop_position(ref_color, from_end=True, until_pos=top)
        right, _ = self.get_crop_position(ref_color, vertical=True, from_end=True, until_pos=left)

        if top >= bottom:
            top, bottom = 0, self.img.shape[0]
        if left >= right:
            left, right = 0, self.img.shape[1]

        return top, left, bottom, right

    def get_cropped_image(self):
        """
        :return: the cropped image, of the same type than the input
        """
        top, left, bottom, right = self.get_cropping_box()
        if isinstance(self.original_img, PILImage):
            return self.original_img.crop((left, top, right, bottom))
        else:
            return self.original_img[top:bottom, left:right]

    def has_crop(self) -> bool:
        return self.get_cropping_box() != (0, 0,) + self.img.shape[:2]

    @auto_plot()
    def show(self, ax=None):
        box = self.get_cropping_box()
        ax.imshow(self.original_img)
        ax.axhline(box[0])
        ax.axvline(box[1])
        ax.axhline(box[2])
        ax.axvline(box[3])
        ax.axis('off')
        ax.set_title(f'Image {self.img.shape}; box {box}; cropped {(box[2] - box[0], box[3] - box[1])}')


def crop_image(*args, **kwargs):
    return ImageCropper(*args, **kwargs).get_cropped_image()


def crop_image_from_file(source_file, target_file):
    img = PIL.Image.open(source_file)
    cropped_img = ImageCropper(img).get_cropped_image()
    cropped_img.save(target_file)
