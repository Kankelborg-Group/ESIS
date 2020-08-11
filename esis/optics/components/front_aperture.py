import typing as typ
import dataclasses
import pandas
import astropy.units as u
from kgpy import Name, optics, format
from . import Component

__all__ = ['FrontAperture']


@dataclasses.dataclass
class FrontAperture(Component):
    name: Name = dataclasses.field(default_factory=lambda: Name('front aperture'))
    piston: u.Quantity = 0 * u.mm
    clear_radius: u.Quantity = 0 * u.mm

    @property
    def surface(self) -> optics.surface.Standard:
        return optics.surface.Standard(
            name=self.name,
            thickness=self.piston,
            aperture=optics.aperture.Circular(
                is_test_stop=False,
                radius=self.clear_radius
            ),
        )

    @property
    def _surfaces(self) -> optics.surface.Standard:
        return self.surface

    def copy(self) -> 'FrontAperture':
        return type(self)(
            piston=self.piston.copy(),
            clear_radius=self.clear_radius.copy(),
            name=self.name.copy(),
        )

    @property
    def dataframe(self) -> pandas.DataFrame:
        return pandas.DataFrame.from_dict(
            data={
                'piston': format.quantity(self.piston),
                'clear radius': format.quantity(self.clear_radius),
            },
            orient='index',
            columns=[str(self.name)],
        )
