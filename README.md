# pyimgy
A small library of image tools for Python

## Features

- `ImageConverter`: a universal, extensible component for easily converting images to different types, array shapes and normalizations.
- `core`: seamless conversion between numpy and Pillow; annotations for conversion and auto plot axes
- `image_crop`: `ImageCropper`, automatic cropping of an image's border frame
- `utils`: various tools
  - get palette of an image

## Installation (in development mode)

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
