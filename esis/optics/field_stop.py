import typing as typ
import dataclasses
import numpy as np
import pandas
import astropy.units as u
import kgpy.labeled
import kgpy.uncertainty
import kgpy.transforms
import kgpy.optics

__all__ = ['FieldStop']

SurfaceT = kgpy.optics.surfaces.Surface[
    None,
    None,
    kgpy.optics.surfaces.apertures.RegularPolygon,
    kgpy.optics.surfaces.apertures.Circular,
    None,
]


@dataclasses.dataclass
class FieldStop(kgpy.optics.components.PistonComponent[SurfaceT]):
    name: str = 'field stop'
    clear_radius: kgpy.uncertainty.ArrayLike = 0 * u.mm
    mech_radius: kgpy.uncertainty.ArrayLike = 0 * u.mm
    num_sides: int = 0

    @property
    def clear_width(self) -> u.Quantity:
        return 2 * self.clear_radius * np.cos(360 * u.deg / self.num_sides / 2)

    @property
    def surface(self) -> SurfaceT:
        surface = super().surface
        surface.is_field_stop = True
        surface.aperture = kgpy.optics.surfaces.apertures.RegularPolygon(
            radius=self.clear_radius,
            num_sides=self.num_sides,
            # offset_angle=180 * u.deg / self.num_sides,
        )
        surface.aperture_mechanical = kgpy.optics.surfaces.apertures.Circular(
            radius=self.mech_radius,
        )
        return surface

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        dataframe['clear radius'] = [format.quantity(self.clear_radius.to(u.mm))]
        dataframe['mechanical radius'] = [format.quantity(self.mech_radius.to(u.mm))]
        dataframe['number of sides'] = [self.num_sides]
        return dataframe
