"""
Package which provides a model of the ESIS instrument and classes for representing and analyzing ESIS observations.
"""
__all__ = [
    'optics', 'Optics',
    'data',
    'flight',
]

from . import optics
from .optics import Optics
from . import data
from . import flight
