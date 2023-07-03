import typing as typ
import dataclasses
import pandas
import astropy.units as u
import kgpy.uncertainty
import kgpy.optics

__all__ = ['FrontAperture']

SurfT = kgpy.optics.surfaces.Surface[
    None,
    None,
    kgpy.optics.surfaces.apertures.Circular,
    None,
    None,
]


@dataclasses.dataclass
class FrontAperture(kgpy.optics.components.PistonComponent[SurfT]):
    name: str = 'front aperture'
    clear_radius: kgpy.uncertainty.ArrayLike = 0 * u.mm

    @property
    def surface(self) -> SurfT:
        surface = super().surface
        # surface.aperture = optics.aperture.Circular(
        #         radius=self.clear_radius
        # )
        return surface

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        dataframe['clear radius'] = [format.quantity(self.clear_radius.to(u.mm))]
        return dataframe
