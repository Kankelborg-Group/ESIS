from dataclasses import dataclass
from kgpy.mixin import Pickleable
from esis.data.inversion.mart import Result
import typing as typ
from astropy import wcs
import pathlib
from kgpy.plot import HypercubeSlicer
import numpy as np
import kgpy.moment
from esis.data.inversion import mart

all = ['default_path','main_event']

default_path = pathlib.Path(__file__).parents[1] / 'flight/level4.pickle'
main_event = pathlib.Path(__file__).parents[1] / 'flight/lev4_mainevent_mart.pickle'

@dataclass
class Level_4(Pickleable):
    cube_list: typ.List[Result]
    wcs_list: typ.List[wcs.WCS]

    @classmethod
    def from_pickle(cls, path: typ.Optional[pathlib.Path] = None) -> 'Level_4':
        obs = super().from_pickle(path)
        for i in range(len(obs.cube_list)):
            obs.wcs_list[i].array_shape = obs.cube_list[i].shape

        return obs

    def plot(self):
        return HypercubeSlicer(self.cube_list,self.wcs_list,(0,99.95),width_ratios=(5, 1),
                height_ratios=(5, 1))

    @property
    def best_inverted_results(self):
        return np.array([self.cube_list[i].best_cube for i in range(len(self.cube_list))])

    @property
    def integrated_intensity(self):
        return [np.sum(img,-1) for img in self.cube_list]

    @property
    def int_wcs(self):
        return [wcs.dropaxis(0) for wcs in self.wcs_list]

    def widths(self,intensity_threshold_percentile = 99.9):
        int = self.integrated_intensity
        threshold = np.percentile(int, intensity_threshold_percentile)

        widths = []
        for i,cube in enumerate(self.cube_list):
            width = np.squeeze(kgpy.moment.percentile.width(cube))
        #   shift = cube.argmax(axis = -1)
            width[int[i] < threshold] = 0
            widths.append(width)

        return widths

    def shifts(self,intensity_threshold_percentile = 99.9):
        int = self.integrated_intensity
        threshold = np.percentile(int,intensity_threshold_percentile)

        shifts = []
        for i,cube in enumerate(self.cube_list):
            shift = np.squeeze(kgpy.moment.percentile.shift(cube)) - cube.shape[-1]//2
        #   shift = cube.argmax(axis = -1)
            shift[int[i] < threshold] = 0
            shifts.append(shift)

        return shifts




