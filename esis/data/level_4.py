from dataclasses import dataclass
from kgpy.mixin import Pickleable
from esis.data.inversion.mart import Result
import typing as typ
from astropy import wcs
import pathlib
from kgpy.plot import HypercubeSlicer
import numpy as np
from esis.data.inversion import mart

all = ['default_path','main_event']

default_path = pathlib.Path(__file__).parents[1] / 'flight/level4.pickle'
main_event = pathlib.Path(__file__).parents[1] / 'flight/lev4_mainevent_mart.pickle'

@dataclass
class Level_4(Pickleable):
    cube_list: typ.List[Result]
    wcs_list: typ.List[wcs.WCS]

    @staticmethod
    def default_pickle_path() -> pathlib.Path:
        return default_path

    @classmethod
    def from_pickle(cls, path: typ.Optional[pathlib.Path] = None) -> 'Level_4':
        obs = super().from_pickle(path)
        return obs

    def plot(self):
        return HypercubeSlicer(self.cube_list,self.wcs_list,(0,99.95),width_ratios=(5, 1),
                height_ratios=(5, 1))

    @property
    def best_inverted_results(self):
        return np.array([self.cube_list[i].best_cube for i in range(len(self.cube_list))])




