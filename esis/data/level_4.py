from dataclasses import dataclass
from kgpy.mixin import Pickleable
from esis.data.inversion.mart import Result
import typing as typ
from astropy import wcs
import pathlib
from kgpy.plot import HypercubeSlicer

all = ['default_path']

default_path = pathlib.Path(__file__).parent / 'level4.pickle'

@dataclass
class Level_4(Pickleable):
    cube_list: typ.List[Result]
    wcs_list: typ.List[wcs.WCS]

    @staticmethod
    def default_pickle_path() -> pathlib.Path:
        return default_path

    def plot(self):
        return HypercubeSlicer(self.cube_list,self.wcs_list)