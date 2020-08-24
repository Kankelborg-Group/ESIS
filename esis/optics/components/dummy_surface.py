import typing as typ
import dataclasses
import astropy.units as u
from kgpy import Name, optics
from . import Component

__all__ = ['DummySurface']


@dataclasses.dataclass
class DummySurface(Component):
    name: Name = dataclasses.field(default_factory=lambda: Name('dummy'))
    thickness: u.Quantity = 100 * u.mm

    @property
    def surface(self) -> optics.surface.CoordinateTransform:
        return optics.surface.CoordinateTransform(
            name=self.name,
            thickness=self.thickness,
        )

    @property
    def _surfaces(self) -> optics.surface.CoordinateTransform:
        return self.surface

    def copy(self) -> 'DummySurface':
        return DummySurface(
            thickness=self.thickness.copy(),
            name=self.name.copy(),
        )
