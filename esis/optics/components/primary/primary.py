import typing as typ
import dataclasses
import numpy as np
import pandas
import astropy.units as u
from kgpy import Name, optics, format

__all__ = ['Primary']

SurfaceT = optics.Surface[
    optics.sag.Standard,
    optics.material.Mirror,
    optics.aperture.RegularPolygon,
    optics.aperture.RegularPolygon,
    None,
]


@dataclasses.dataclass
class Primary(optics.component.PistonComponent[SurfaceT]):
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
    def surface(self) -> SurfaceT:
        surface = super().surface  # type: SurfaceT
        surface.sag = optics.sag.Standard(
            radius=-self.radius,
            conic=self.conic,
        )
        surface.material = optics.material.Mirror(thickness=self.substrate_thickness)
        surface.aperture = optics.aperture.RegularPolygon(
            radius=self.clear_radius,
            num_sides=self.num_sides,
            offset_angle=180 * u.deg / self.num_sides,
        )
        surface.aperture_mechanical = optics.aperture.RegularPolygon(
            radius=self.clear_radius + self.border_width,
            num_sides=self.num_sides,
            offset_angle=180 * u.deg / self.num_sides,
        )
        return surface

    def copy(self) -> 'Primary':
        other = super().copy()      # type: Primary
        other.radius = self.radius.copy()
        other.conic = self.conic.copy()
        other.num_sides = self.num_sides
        other.clear_radius = self.clear_radius.copy()
        other.border_width = self.border_width.copy()
        other.substrate_thickness = self.substrate_thickness.copy()
        return other

    @property
    def dataframe(self) -> pandas:
        dataframe = super().dataframe
        dataframe['radius'] = [format.quantity(self.radius.to(u.mm))]
        dataframe['conic constant'] = [format.quantity(self.conic)]
        dataframe['number of sides'] = [self.num_sides]
        dataframe['clear radius'] = [format.quantity(self.clear_radius.to(u.mm))]
        dataframe['border width'] = [format.quantity(self.border_width.to(u.mm))]
        dataframe['substrate thickness'] = [format.quantity(self.substrate_thickness.to(u.mm))]
        return dataframe
