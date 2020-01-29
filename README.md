# pyimgy
A small library of image tools for Python

## Features

- `ImageConverter`: a universal, extensible component for easily converting images to different types, array shapes and normalizations.
- `core`: seamless conversion between numpy and Pillow; annotations for conversion and auto plot axes
- `image_crop`: `ImageCropper`, automatic cropping of an image's border frame
- `utils`: various tools
  - get palette of an image

## Overview of features

### `convert_image`: universal image converter

This function allows to convert a given image data (which could correspond to one image or a batch), from various Python types, shapes and value
normalization. Basic usage:

```python
from pyimgy.core import *

img = PIL.Image.open('images/bedroom-with-border.jpg')

converted_img = convert_image(img, to_type=np.ndarray, shape='3WH', norm='int_255')
```

The arguments `to_type`, `shape` and `norm` are all optional, but at least one must be given.

- `to_type` indicates the Python type to which to convert the image (i.e. the type of the output). Possible values so far (from and to):
  - `numpy.ndarray`
  - `PIL.Image.Image` (the Image class from Pillow)
  - `torch.Tensor` (optional, if PyTorch is installed)
  - `fast.vision.Image` (optional, if fast.ai is installed)
- `shape` is used to re-shape the image data to one of the standard representations, given by a key. The number of character in the key tells how many
dimensions the output shape should have, and the characters themselves determine what each dimension is.
  - 2 dim: `WH` (width, height)
  - 3 dim: `1WH, 3WH, CWH, WH1, WH3, WHC` (the color channel can have size 1 or 3, or `C` as a wildcard, and can be leading or trailing)
  - 4 dim: `13WH, 11WH, 1CWH, 1WH3` (the first dimension is always the batch size)
- `norm` to specify a normalization for the data itself
  - `int_255`: integer, scaled so the max value is 255
  - `float_1`: float, scaled so the max value is 1.0
  
### `@converting`: annotation for the image converter

It is useful to abstract out the image conversion logic from functions. See the doc info in [core](pyimgy/core.py).

### `@auto_plot`: annotation for clean an easy plotting configuration

This very handy annotation can be used in regular Python modules and Jupyter notebooks. See the doc and examples in [core](pyimgy/core.py).

### `ImageCropper`: remove borders from images

Component to automatically remove border frames of the same color from an image, it is robust to small variations in the color.
Code in [image_crop](pyimgy/image_crop.py).

### Various image utils

They are in [utils](pyimgy/utils.py). Some of them are:
- `get_image_palette`:  For the given image, return a palette of the request size, with the optimal colors. It leverages Pillow's palette conversion.
- `get_color_distribution`: Return the unique colors detected in a given image, and how many pixels each covers (absolute or a ratio)

## Video utils

Some functions to easily extract frames from videos. See [video_utils](pyimgy/video/video_utils.py).

## Installing dependencies

This project uses [Pipenv](https://pipenv-fork.readthedocs.io/en/latest/index.html) to manage dependencies. They are specified in the `Pipfile`.

In order to install the dependencies in the local environment, run: `pipenv install`

If you want to open a shell in the virtual environment, run: `pipenv shell`

## Tests

In order to execute the provided tests, run: `python -m unittest` from the repositories folder (you might need to be in the virtualenv).

## Installation (locally, in development mode)

To install the standard functionalities, run this from the repository's directory:

```
pip install -e .
```

To install the "optional" modules for Pytorch and fast.ai, run the following:

```
pip install -e '.[torch]'
pip install -e '.[fastai]'
```

You might need to restart your IDE afterwards.

## Installing from PyPI

This project is in [PyPI](https://pypi.org/project/pyimgy/). Install it with `pip install pyimgy` 

## Usage examples

Check out our cool Deep Learning visualization project at [deepfx/netlens](https://github.com/deepfx/netlens).
