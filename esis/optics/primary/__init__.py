import typing as typ
import dataclasses
import numpy as np
import pandas
import astropy.units as u
import kgpy.mixin
import kgpy.format
import kgpy.labeled
import kgpy.uncertainty
import kgpy.optics
from . import efficiency

__all__ = ['Primary']

SurfaceT = kgpy.optics.surfaces.Surface[
    kgpy.optics.surfaces.sags.Standard,
    kgpy.optics.surfaces.materials.Mirror,
    kgpy.optics.surfaces.apertures.RegularPolygon,
    kgpy.optics.surfaces.apertures.RegularPolygon,
    None,
]

#
# class PrimaryAxes(mixin.AutoAxis):
#     def __init__(self):
#         super().__init__()
#         self.primary_translation_x = self.auto_axis_index(from_right=False)
#         self.primary_translation_y = self.auto_axis_index(from_right=False)
#         self.primary_translation_z = self.auto_axis_index(from_right=False)


@dataclasses.dataclass
class Primary(kgpy.optics.components.TranslationComponent[SurfaceT]):
    name: str = 'primary'
    radius: kgpy.uncertainty.ArrayLike = np.inf * u.mm
    conic: kgpy.uncertainty.ArrayLike = -1 * u.dimensionless_unscaled
    mtf_degradation_factor: kgpy.uncertainty.ArrayLike = 0 * u.dimensionless_unscaled
    slope_error: kgpy.optics.surfaces.sags.SlopeErrorRMS = dataclasses.field(default_factory=kgpy.optics.surfaces.sags.SlopeErrorRMS)
    ripple: kgpy.optics.surfaces.sags.RippleRMS = dataclasses.field(default_factory=kgpy.optics.surfaces.sags.RippleRMS)
    microroughness: kgpy.optics.surfaces.sags.RoughnessRMS = dataclasses.field(default_factory=kgpy.optics.surfaces.sags.RoughnessRMS)
    num_sides: int = 0
    clear_half_width: kgpy.uncertainty.ArrayLike = 0 * u.mm
    border_width: kgpy.uncertainty.ArrayLike = 0 * u.mm
    material: kgpy.optics.surfaces.materials.MultilayerMirror = dataclasses.field(
        default_factory=kgpy.optics.surfaces.materials.MultilayerMirror)

    @property
    def focal_length(self) -> kgpy.uncertainty.ArrayLike:
        return self.radius / 2

    @property
    def clear_radius(self) -> kgpy.uncertainty.ArrayLike:
        return self.clear_half_width / np.cos(360 * u.deg / self.num_sides / 2)

    @property
    def mech_radius(self) -> kgpy.uncertainty.ArrayLike:
        return (self.clear_half_width + self.border_width) / np.cos(360 * u.deg / self.num_sides / 2)

    @property
    def mech_half_width(self) -> kgpy.uncertainty.ArrayLike:
        return self.clear_half_width + self.border_width

    @property
    def surface(self) -> SurfaceT:
        surface = super().surface  # type: SurfaceT
        surface.sag = kgpy.optics.surfaces.sags.Standard(
            radius=-self.radius,
            conic=self.conic,
        )
        surface.material = self.material.copy()
        surface.aperture = kgpy.optics.surfaces.apertures.RegularPolygon(
            radius=self.clear_radius,
            num_sides=self.num_sides,
            # offset_angle=180 * u.deg / self.num_sides,
        )
        surface.aperture_mechanical = kgpy.optics.surfaces.apertures.RegularPolygon(
            radius=self.mech_radius,
            num_sides=self.num_sides,
            # offset_angle=180 * u.deg / self.num_sides,
        )
        return surface

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        dataframe['radius'] = [kgpy.format.quantity(self.radius.to(u.mm))]
        dataframe['conic constant'] = [kgpy.format.quantity(self.conic)]
        dataframe['slope error'] = [kgpy.format.quantity(self.slope_error.value)]
        dataframe['ripple'] = [kgpy.format.quantity(self.ripple.value)]
        dataframe['microroughness'] = [kgpy.format.quantity(self.microroughness.value)]
        dataframe['number of sides'] = [self.num_sides]
        dataframe['clear half-width'] = [kgpy.format.quantity(self.clear_half_width.to(u.mm))]
        dataframe['border width'] = [kgpy.format.quantity(self.border_width.to(u.mm))]
        dataframe['substrate thickness'] = [kgpy.format.quantity(self.material.thickness.to(u.mm))]
        return dataframe
