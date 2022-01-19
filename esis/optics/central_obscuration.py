import typing as typ
import dataclasses
import numpy as np
import pandas
import astropy.units as u
from kgpy import Name, optics, format, transform, vector

__all__ = ['CentralObscuration']

SurfaceT = optics.surface.Surface[None, None, optics.surface.aperture.RegularPolygon, None, None]


@dataclasses.dataclass
class CentralObscuration(optics.component.PistonComponent[SurfaceT]):
    name: Name = dataclasses.field(default_factory=lambda: Name('obscuration'))
    obscured_half_width: u.Quantity = 0 * u.mm
    position_error: vector.Vector3D = dataclasses.field(default_factory=vector.Vector3D.spatial)

    @property
    def obscured_radius(self) -> u.Quantity:
        return self.obscured_half_width / np.cos(360 * u.deg / 8 / 2)

    @property
    def surface(self) -> SurfaceT:
        surface = super().surface  # type: SurfaceT
        # offset_angle = 360 * u.deg / 8 / 2
        offset_angle = 360 * u.deg / 8
        angles = np.linspace(0, 360 * u.deg, 8, endpoint=False)[:~0] - offset_angle
        surface.aperture = optics.surface.aperture.IrregularPolygon(
        # surface.aperture = optics.aperture.RegularPolygon(
            is_obscuration=True,
            decenter=transform.rigid.Translate.from_vector(self.position_error),
            vertices=vector.Vector3D.from_cylindrical(radius=self.obscured_radius, azimuth=angles, z=0*u.mm),
            # radius=self.obscured_radius,
            # num_sides=self.num_sides,
            # offset_angle=360 * u.deg / self.num_sides / 2,
        )
        return surface

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        dataframe['obscured half-width'] = [format.quantity(self.obscured_half_width.to(u.mm))]
        # dataframe['number of sides'] = [self.num_sides]
        return dataframe
