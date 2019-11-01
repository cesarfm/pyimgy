import PIL.Image

from pyimgy.image_crop import *
from pyimgy.utils import *


def main():
    img = PIL.Image.open('../images/bedroom-with-border.jpg')
    cropper = ImageCropper(img)
    print(cropper.get_cropping_box())
    cropper.show()

    show_image_palette(img, 8)


if __name__ == '__main__':
    main()
