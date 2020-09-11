"""
Package is for code and information relating to the 2019 flight
"""
__all__ = ['raw_img_dir', 'level_0']

import pathlib
from .. import data
raw_img_dir = pathlib.Path(__file__).parent / 'images'


def level_0():
    return data.Level_0.from_directory(directory=raw_img_dir)
