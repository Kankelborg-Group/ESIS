import typing as typ
import dataclasses
import astropy.units as u
from kgpy import Name, optics
from . import Component

__all__ = ['FrontAperture']


@dataclasses.dataclass
class FrontAperture(Component):
    name: Name = dataclasses.field(default_factory=lambda: Name('front_aperture'))
    piston: u.Quantity = 0 * u.mm
    clear_radius: u.Quantity = 0 * u.mm

    @property
    def surface(self) -> optics.surface.Standard:
        return optics.surface.Standard(
            name=self.name,
            thickness=-self.piston,
            aperture=optics.aperture.Circular(
                is_test_stop=False,
                radius=self.clear_radius
            ),
        )

    @property
    def _surfaces(self) -> optics.surface.Standard:
        return self.surface

    def copy(self) -> 'FrontAperture':
        return FrontAperture(
            piston=self.piston.copy(),
            clear_radius=self.clear_radius.copy(),
            name=self.name.copy(),
        )
