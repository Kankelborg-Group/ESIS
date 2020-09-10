"""
Package is for code and information relating to the 2019 flight
"""
__all__ = ['raw_img_dir']

import pathlib
raw_img_dir = pathlib.Path(__file__).parent / 'images'
