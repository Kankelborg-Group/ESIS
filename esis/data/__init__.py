"""
Represent and process ESIS observations into spatial-spectral cubes.

Description of the ESIS Data Levels
===================================

Level 0
-------

* The raw data gathered by the ESIS instrument, saved as FITS files.

"""

from ._level_0 import Level_0

__all__ = [
    "Level_0",
]
