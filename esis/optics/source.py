import typing
import dataclasses
import astropy.units as u
import pandas
from kgpy import Name, format, transform, optics

__all__ = ['Source']

SurfaceT = optics.Surface[None, None, optics.aperture.Rectangular, None, None]


@dataclasses.dataclass
class Source(optics.component.PistonComponent):
    name: Name = dataclasses.field(default_factory=lambda: Name('sun'))
    half_width_x: u.Quantity = 0 * u.deg
    half_width_y: u.Quantity = 0 * u.deg

    @property
    def surface(self) -> SurfaceT:
        surface = super().surface   # type: SurfaceT
        surface.aperture = optics.aperture.Rectangular(
            decenter=transform.rigid.Translate([0, 0, 0] * u.deg),
            half_width_x=self.half_width_x,
            half_width_y=self.half_width_y,
        )
        return surface

    def copy(self) -> 'Source':
        other = super().copy()  # type: Source
        other.half_width_x = self.half_width_x.copy()
        other.half_width_y = self.half_width_y.copy()
        return other

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        dataframe['half width'] = [format.quantity(self.half_width_x.to(u.arcmin))]
        dataframe['half height'] = [format.quantity(self.half_width_y.to(u.arcmin))]
        return dataframe
