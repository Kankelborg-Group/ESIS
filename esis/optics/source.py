import typing
import dataclasses
import astropy.units as u
import pandas
from kgpy import Name, vector, format, transform, optics

__all__ = ['Source']

SurfaceT = optics.surface.Surface[None, None, optics.surface.aperture.Rectangular, None, None]


@dataclasses.dataclass
class Source(optics.component.PistonComponent):
    name: Name = dataclasses.field(default_factory=lambda: Name('sun'))
    decenter: vector.Vector2D = dataclasses.field(default_factory=vector.Vector2D.angular)
    half_width_x: u.Quantity = 0 * u.deg
    half_width_y: u.Quantity = 0 * u.deg

    @property
    def surface(self) -> SurfaceT:
        surface = super().surface  # type: SurfaceT
        surface.aperture = optics.surface.aperture.Rectangular(
            decenter=transform.rigid.Translate(x=self.decenter.x, y=self.decenter.y, z=0 * u.deg),
            half_width_x=self.half_width_x,
            half_width_y=self.half_width_y,
        )
        return surface

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        dataframe['half width'] = [format.quantity(self.half_width_x.to(u.arcmin))]
        dataframe['half height'] = [format.quantity(self.half_width_y.to(u.arcmin))]
        return dataframe
