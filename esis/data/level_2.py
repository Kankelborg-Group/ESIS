import typing as typ
import dataclasses
import astropy.units as u
import kgpy.obs
import esis
from . import Level_1

__all__ = ['Level_2']


@dataclasses.dataclass
class Level_2(kgpy.obs.Image):
    spikes: typ.Optional[u.Quantity] = None
    optics: typ.Optional[esis.optics.Optics] = None

    @classmethod
    def from_level_1(cls, level_1: Level_1):
        pass

    @property
    def intensity_respike(self):
        return self.intensity + self.spikes
