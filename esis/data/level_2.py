import dataclasses
import typing as typ
from kgpy import obs, mixin

__all__ = ['Level_2']


@dataclasses.dataclass
class Level_2(obs.Image, mixin.Pickleable):
    pass
