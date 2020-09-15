"""
Package is for code and information relating to the 2019 flight
"""
__all__ = [
    'optics',
    'raw_img_dir',
    'level_0'
]

import pathlib
from .. import data
from . import optics

raw_img_dir = pathlib.Path(__file__).parent / 'images'
num_dark_safety_frames = 3


def level_0(caching: bool = False):
    return data.Level_0.from_directory(
        directory=raw_img_dir,
        detector=optics.as_measured().detector,
        caching=caching,
        num_dark_safety_frames=num_dark_safety_frames,
    )
