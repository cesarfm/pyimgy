#!/usr/bin/env python
"""
# pyimgy
A small library of image tools for Python

## Features

- `ImageConverter`: a universal, extensible component for easily converting images to different types, array shapes and normalizations.
- `core`: seamless conversion between numpy and Pillow; annotations for conversion and auto plot axes
- `image_crop`: `ImageCropper`, automatic cropping of an image's border frame
- `utils`: various tools
  - get palette of an image
"""

from setuptools import setup, find_packages

DOCLINES = (__doc__ or '').split("\n")
long_description = "\n".join(DOCLINES[2:])

version = '0.1.0'

setup(
    name='pyimgy',
    version=version,
    author='César Fuentes',
    author_email='cesar.at.fuentes@gmail.com',
    description='A small library of image tools for Python',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/cesarfm/pyimgy',
    packages=find_packages(include=['pyimgy', 'pyimgy.*']),
    install_requires=[
        'typing',
        'numpy',
        'Pillow',
        'matplotlib',
        'opencv-python'
    ],
    extras_require={
        'torch': ['torch'],
        'fastai': ['torch', 'fastai']
    },
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
