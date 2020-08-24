import typing as typ
import dataclasses
import numpy as np
import pandas
from kgpy.mixin import PandasDataframable
import astropy.units as u
from kgpy import Name, optics, format
from .. import Component

__all__ = ['Primary']

AperSurfT = optics.surface.Standard[None, optics.aperture.RegularPolygon]
MainSurfT = optics.surface.Standard[optics.material.Mirror, optics.aperture.RegularPolygon]


@dataclasses.dataclass
class Primary(Component, PandasDataframable):
    name: Name = dataclasses.field(default_factory=lambda: Name('primary'))
    radius: u.Quantity = np.inf * u.mm
    conic: u.Quantity = -1 * u.dimensionless_unscaled
    num_sides: int = 0
    clear_radius: u.Quantity = 0 * u.mm
    border_width: u.Quantity = 0 * u.mm
    substrate_thickness: u.Quantity = 0 * u.mm

    @property
    def focal_length(self) -> u.Quantity:
        return self.radius / 2

    @property
    def surface(self) -> AperSurfT:
        return optics.surface.Standard(
            name=self.name + 'aper',
            radius=-self.radius,
            conic=self.conic,
            aperture=optics.aperture.RegularPolygon(
                is_test_stop=False,
                radius=self.clear_radius,
                num_sides=self.num_sides,
                offset_angle=180 * u.deg / self.num_sides,
            ),
        )

    @property
    def main_surface(self) -> MainSurfT:
        return optics.surface.Standard(
            name=self.name + 'main',
            radius=-self.radius,
            conic=self.conic,
            material=optics.material.Mirror(thickness=self.substrate_thickness),
            aperture=optics.aperture.RegularPolygon(
                is_active=False,
                radius=self.clear_radius + self.border_width,
                num_sides=self.num_sides,
                offset_angle=180 * u.deg / self.num_sides,
            ),
        )

    @property
    def _surfaces(self) -> optics.surface.Transformed[optics.surface.Substrate[AperSurfT, MainSurfT]]:
        return optics.surface.Transformed(
            name=self.name,
            surfaces=optics.surface.Substrate(
                aperture_surface=self.surface,
                main_surface=self.main_surface,
            ),
        )

    def copy(self) -> 'Primary':
        return Primary(
            radius=self.radius.copy(),
            num_sides=self.num_sides,
            clear_radius=self.clear_radius.copy(),
            border_width=self.border_width.copy(),
            substrate_thickness=self.substrate_thickness.copy(),
            name=self.name.copy(),
            conic=self.conic
        )

    @property
    def dataframe(self) -> pandas:
        return pandas.DataFrame.from_dict(
            data={
                'radius': format.quantity(self.radius.to(u.mm)),
                'conic constant': format.quantity(self.conic),
                'number of sides': self.num_sides,
                'clear radius': format.quantity(self.clear_radius.to(u.mm)),
                'border width': format.quantity(self.border_width.to(u.mm)),
                'substrate thickness': format.quantity(self.substrate_thickness.to(u.mm)),
            },
            orient='index',
            columns=[str(self.name)],
        )
