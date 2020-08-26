import typing as typ
import dataclasses
import pandas
import astropy.units as u
from kgpy import Name, transform, optics, format

__all__ = ['FieldStop']

SurfT = optics.surface.Standard[None, optics.aperture.RegularPolygon, optics.aperture.Circular]


@dataclasses.dataclass
class FieldStop(optics.component.PistonComponent[SurfT]):
    name: Name = dataclasses.field(default_factory=lambda: Name('field stop'))
    clear_radius: u.Quantity = 0 * u.mm
    mech_radius: u.Quantity = 0 * u.mm
    num_sides: int = 0

    @property
    def surface(self) -> SurfT:
        surface = super().surface
        surface.aperture = optics.aperture.RegularPolygon(
            radius=self.clear_radius,
            num_sides=self.num_sides,
            offset_angle=180 * u.deg / self.num_sides,
        )
        surface.aperture_mechanical = optics.aperture.Circular(
            radius=self.mech_radius,
        )
        return surface

    def copy(self) -> 'FieldStop':
        other = super().copy()      # type: FieldStop
        other.clear_radius = self.clear_radius.copy()
        other.mech_radius = self.mech_radius.copy()
        other.num_sides = self.num_sides
        return other

    @property
    def dataframe(self) -> pandas.DataFrame:
        return pandas.DataFrame.from_dict(
            data={
                'piston': format.quantity(self.piston.to(u.mm)),
                'clear radius': format.quantity(self.clear_radius.to(u.mm)),
                'mechanical radius': format.quantity(self.mech_radius.to(u.mm)),
                'number of sides': self.num_sides,
            },
            orient='index',
            columns=[str(self.name)],
        )
