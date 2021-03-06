import typing as typ
import dataclasses
import numpy as np
import pandas
import astropy.units as u
from kgpy import Name, transform, optics, format

__all__ = ['Filter']

SurfT = optics.surface.Surface[None, None, optics.surface.aperture.Circular, optics.surface.aperture.Circular, None]


@dataclasses.dataclass
class Filter(optics.component.CylindricalComponent[SurfT]):
    name: Name = dataclasses.field(default_factory=lambda: Name('filter'))
    inclination: u.Quantity = 0 * u.deg
    clear_radius: u.Quantity = 0 * u.mm
    border_width: u.Quantity = 0 * u.mm
    thickness: u.Quantity = 0 * u.mm
    thickness_oxide: u.Quantity = 0 * u.mm
    mesh_ratio: u.Quantity = 100 * u.percent

    @property
    def clear_diameter(self) -> u.Quantity:
        return 2 * self.clear_radius

    @property
    def transform(self) -> transform.rigid.TransformList:
        return super().transform + transform.rigid.TransformList([
            transform.rigid.TiltY(-self.inclination)
        ])

    @property
    def surface(self) -> SurfT:
        surface = super().surface
        surface.aperture = optics.surface.aperture.Circular(
            radius=self.clear_radius
        )
        surface.aperture_mechanical = optics.surface.aperture.Circular(
            radius=self.clear_radius + self.border_width,
        )
        surface.material = optics.surface.material.AluminumThinFilm(
            thickness=self.thickness,
            mesh_ratio=self.mesh_ratio,
        )
        return surface

    def copy(self) -> 'Filter':
        other = super().copy()      # type: Filter
        other.inclination = self.inclination.copy()
        other.clear_radius = self.clear_radius.copy()
        other.border_width = self.border_width.copy()
        other.thickness = self.thickness.copy()
        other.thickness_oxide = self.thickness_oxide.copy()
        other.mesh_ratio = self.mesh_ratio.copy()
        return other

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        dataframe['inclination'] = [format.quantity(self.inclination.to(u.deg))]
        dataframe['clear radius'] = [format.quantity(self.clear_radius.to(u.mm))]
        dataframe['border width'] = [format.quantity(self.border_width.to(u.mm))]
        dataframe['thickness'] = [format.quantity(self.thickness.to(u.nm))]
        dataframe['oxide thickness'] = [format.quantity(self.thickness_oxide.to(u.nm))]
        dataframe['mesh ratio'] = [format.quantity(self.mesh_ratio.to(u.percent))]
        return dataframe
