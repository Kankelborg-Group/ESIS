import typing as typ
import dataclasses
import numpy as np
import pandas
import astropy.units as u
import kgpy.format
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vectors
import kgpy.transforms
import kgpy.optics

__all__ = ['CentralObscuration']

SurfaceT = kgpy.optics.surfaces.Surface[None, None, kgpy.optics.surfaces.apertures.IrregularPolygon, None, None]


@dataclasses.dataclass
class CentralObscuration(kgpy.optics.components.PistonComponent[SurfaceT]):
    name: str = 'obscuration'
    obscured_half_width: kgpy.uncertainty.ArrayLike = 0 * u.mm

    @property
    def obscured_radius(self) -> kgpy.uncertainty.ArrayLike:
        return self.obscured_half_width / np.cos(360 * u.deg / 8 / 2)

    @property
    def surface(self) -> SurfaceT:
        surface = super().surface  # type: SurfaceT
        # offset_angle = 360 * u.deg / 8 / 2
        offset_angle = 360 * u.deg / 8
        angles = kgpy.labeled.LinearSpace(0, 360 * u.deg, 8, endpoint=False, axis='vertex')[dict(vertex=slice(None, ~0))]
        angles = angles - offset_angle
        surface.aperture = kgpy.optics.surfaces.apertures.IrregularPolygon(
        # surface.aperture = optics.aperture.RegularPolygon(
            is_obscuration=True,
            vertices=kgpy.vectors.Cylindrical(radius=self.obscured_radius, azimuth=angles, z=0*u.mm).cartesian,
            # radius=self.obscured_radius,
            # num_sides=self.num_sides,
            # offset_angle=360 * u.deg / self.num_sides / 2,
        )
        return surface

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        dataframe['obscured half-width'] = [kgpy.format.quantity(self.obscured_half_width.to(u.mm))]
        # dataframe['number of sides'] = [self.num_sides]
        return dataframe
