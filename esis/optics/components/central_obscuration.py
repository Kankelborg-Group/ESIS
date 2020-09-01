import typing as typ
import dataclasses
import pandas
import astropy.units as u
from kgpy import Name, optics, format, transform

__all__ = ['CentralObscuration']

SurfT = optics.Surface[None, None, optics.aperture.RegularPolygon, None, None]


@dataclasses.dataclass
class CentralObscuration(optics.component.PistonComponent[SurfT]):
    name: Name = dataclasses.field(default_factory=lambda: Name('obscuration'))
    obscured_radius: u.Quantity = 0 * u.mm
    num_sides: int = 0

    @property
    def surface(self) -> SurfT:
        surface = super().surface  # type: SurfT
        surface.aperture = optics.aperture.RegularPolygon(
            is_obscuration=True,
            radius=self.obscured_radius,
            num_sides=self.num_sides,
            offset_angle=180 * u.deg / self.num_sides,
        )
        return surface

    def copy(self) -> 'CentralObscuration':
        other = super().copy()      # type: CentralObscuration
        other.obscured_radius = self.obscured_radius.copy()
        other.num_sides = self.num_sides
        return other

    @property
    def dataframe(self) -> pandas.DataFrame:
        return pandas.DataFrame.from_dict(
            data={
                'piston': format.quantity(self.piston.to(u.mm)),
                'obscured radius': format.quantity(self.obscured_radius.to(u.mm)),
                'number of sides': self.num_sides,
            },
            orient='index',
            columns=[str(self.name)],
        )
