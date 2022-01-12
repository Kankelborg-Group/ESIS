import typing as typ
import dataclasses
import pandas
import astropy.units as u
from kgpy import Name, optics, format

__all__ = ['FrontAperture']

SurfT = optics.surface.Surface[None, None, optics.surface.aperture.Circular, None, None]


@dataclasses.dataclass
class FrontAperture(optics.component.PistonComponent[SurfT]):
    name: Name = dataclasses.field(default_factory=lambda: Name('front aperture'))
    clear_radius: u.Quantity = 0 * u.mm

    @property
    def surface(self) -> SurfT:
        surface = super().surface
        # surface.aperture = optics.aperture.Circular(
        #         radius=self.clear_radius
        # )
        return surface

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        dataframe['clear radius'] = [format.quantity(self.clear_radius.to(u.mm))]
        return dataframe
