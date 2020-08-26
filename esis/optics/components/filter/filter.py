import typing as typ
import dataclasses
import numpy as np
import pandas
import astropy.units as u
from kgpy import Name, transform, optics, format

__all__ = ['Filter']

SurfT = optics.surface.Standard[None, optics.aperture.Circular, optics.aperture.Circular]


@dataclasses.dataclass
class Filter(optics.component.CylindricalComponent[SurfT]):
    name: Name = dataclasses.field(default_factory=lambda: Name('filter'))
    inclination: u.Quantity = 0 * u.deg
    clear_radius: u.Quantity = 0 * u.mm
    border_width: u.Quantity = 0 * u.mm

    @property
    def transform(self) -> transform.rigid.TransformList:
        return super().transform + transform.rigid.TransformList([
            transform.rigid.TiltY(-self.inclination)
        ])

    @property
    def surface(self) -> SurfT:
        surface = super().surface
        surface.aperture = optics.aperture.Circular(
            radius=self.clear_radius
        )
        surface.aperture_mechanical = optics.aperture.Circular(
            radius=self.clear_radius + self.border_width,
        )
        return surface

    def copy(self) -> 'Filter':
        other = super().copy()      # type: Filter
        other.inclination = self.inclination.copy()
        other.clear_radius = self.clear_radius.copy()
        other.border_width = self.border_width.copy()
        return other

    @property
    def dataframe(self) -> pandas.DataFrame:
        return pandas.DataFrame.from_dict(
            data={
                'piston': format.quantity(self.piston.to(u.mm)),
                'channel radius': format.quantity(self.channel_radius.to(u.mm)),
                'channel angle': format.quantity(self.channel_angle.to(u.deg)),
                'inclination': format.quantity(self.inclination.to(u.deg)),
                'clear radius': format.quantity(self.clear_radius.to(u.mm)),
                'border width': format.quantity(self.border_width.to(u.mm)),
            },
            orient='index',
            columns=[str(self.name)],
        )
