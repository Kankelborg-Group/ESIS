import typing as typ
import dataclasses
import numpy as np
import pandas
import astropy.units as u
from kgpy import Name, optics, format, transform

__all__ = ['CentralObscuration']

SurfaceT = optics.Surface[None, None, optics.aperture.RegularPolygon, None, None]


@dataclasses.dataclass
class CentralObscuration(optics.component.PistonComponent[SurfaceT]):
    name: Name = dataclasses.field(default_factory=lambda: Name('obscuration'))
    obscured_half_width: u.Quantity = 0 * u.mm
    num_sides: int = 0

    @property
    def obscured_radius(self) -> u.Quantity:
        return self.obscured_half_width / np.cos(360 * u.deg / self.num_sides / 2)

    @property
    def surface(self) -> SurfaceT:
        surface = super().surface  # type: SurfaceT
        surface.aperture = optics.aperture.RegularPolygon(
            is_obscuration=True,
            radius=self.obscured_radius,
            num_sides=self.num_sides,
            offset_angle=360 * u.deg / self.num_sides / 2,
        )
        return surface

    def copy(self) -> 'CentralObscuration':
        other = super().copy()      # type: CentralObscuration
        other.obscured_half_width = self.obscured_half_width.copy()
        other.num_sides = self.num_sides
        return other

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        dataframe['obscured half-width'] = [format.quantity(self.obscured_half_width.to(u.mm))]
        dataframe['number of sides'] = [self.num_sides]
        return dataframe
