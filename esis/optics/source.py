import typing
import dataclasses
import astropy.units as u
import pandas
import kgpy.uncertainty
import kgpy.vectors
import kgpy.transforms
import kgpy.optics

__all__ = ['Source']

SurfaceT = kgpy.optics.surfaces.Surface[
    None,
    None,
    kgpy.optics.surfaces.apertures.Rectangular,
    None,
    None,
]


@dataclasses.dataclass
class Source(kgpy.optics.components.PistonComponent):
    name: str = 'sun'
