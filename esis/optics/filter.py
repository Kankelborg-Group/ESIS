import typing as typ
import dataclasses
import numpy as np
import pandas
import astropy.units as u
import kgpy.units
import kgpy.labeled
import kgpy.uncertainty
import kgpy.transforms
import kgpy.optics

__all__ = ['Filter']

SurfT = kgpy.optics.surfaces.Surface[
    None,
    None,
    kgpy.optics.surfaces.apertures.Circular,
    kgpy.optics.surfaces.apertures.Circular,
    None,
]


@dataclasses.dataclass
class Filter(kgpy.optics.components.CylindricalComponent[SurfT]):
    name: str = 'filter'
    inclination: kgpy.uncertainty.ArrayLike = 0 * u.deg
    clocking: kgpy.uncertainty.ArrayLike = 0 * u.deg
    clear_radius: kgpy.uncertainty.ArrayLike = 0 * u.mm
    border_width: kgpy.uncertainty.ArrayLike = 0 * u.mm
    thickness: kgpy.uncertainty.ArrayLike = 0 * u.mm
    thickness_oxide: kgpy.uncertainty.ArrayLike = 0 * u.mm
    mesh_ratio: kgpy.uncertainty.ArrayLike = 100 * u.percent
    mesh_pitch: kgpy.uncertainty.ArrayLike= 0 * kgpy.units.line / u.imperial.inch
    mesh_material: str = ''

    @property
    def clear_diameter(self) -> kgpy.uncertainty.ArrayLike:
        return 2 * self.clear_radius

    @property
    def transform(self) -> kgpy.transforms.TransformList:
        return super().transform + kgpy.transforms.TransformList([
            kgpy.transforms.RotationY(-self.inclination)
        ])

    @property
    def surface(self) -> SurfT:
        surface = super().surface
        surface.aperture = kgpy.optics.surfaces.apertures.Circular(
            is_active=False,
            radius=self.clear_radius
        )
        surface.aperture_mechanical = kgpy.optics.surfaces.apertures.Circular(
            radius=self.clear_radius + self.border_width,
        )
        surface.material = kgpy.optics.surfaces.materials.AluminumThinFilm(
            thickness=self.thickness,
            thickness_oxide=self.thickness_oxide,
            mesh_ratio=self.mesh_ratio,
        )
        return surface

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        dataframe['inclination'] = [format.quantity(self.inclination.to(u.deg))]
        dataframe['clocking'] = [format.quantity(self.clocking.to(u.deg))]
        dataframe['clear radius'] = [format.quantity(self.clear_radius.to(u.mm))]
        dataframe['border width'] = [format.quantity(self.border_width.to(u.mm))]
        dataframe['thickness'] = [format.quantity(self.thickness.to(u.nm))]
        dataframe['oxide thickness'] = [format.quantity(self.thickness_oxide.to(u.nm))]
        dataframe['mesh ratio'] = [format.quantity(self.mesh_ratio.to(u.percent))]
        dataframe['mesh pitch'] = [format.quantity(self.mesh_pitch)]
        return dataframe
