import typing as typ
import dataclasses
import numpy as np
import pandas
import astropy.units as u
from kgpy import Name, optics, format, mixin
from . import efficiency

__all__ = ['Primary']

SurfaceT = optics.surface.Surface[
    optics.surface.sag.Standard,
    optics.surface.material.Mirror,
    optics.surface.aperture.RegularPolygon,
    optics.surface.aperture.RegularPolygon,
    None,
]


class PrimaryAxes(mixin.AutoAxis):
    def __init__(self):
        super().__init__()
        self.primary_translation_x = self.auto_axis_index(from_right=False)
        self.primary_translation_y = self.auto_axis_index(from_right=False)
        self.primary_translation_z = self.auto_axis_index(from_right=False)


@dataclasses.dataclass
class Primary(optics.component.TranslationComponent[SurfaceT]):
    name: Name = dataclasses.field(default_factory=lambda: Name('primary'))
    radius: u.Quantity = np.inf * u.mm
    conic: u.Quantity = -1 * u.dimensionless_unscaled
    mtf_degradation_factor: u.Quantity = 0 * u.dimensionless_unscaled
    slope_error: optics.surface.sag.SlopeErrorRMS = dataclasses.field(default_factory=optics.surface.sag.SlopeErrorRMS)
    ripple: optics.surface.sag.RippleRMS = dataclasses.field(default_factory=optics.surface.sag.RippleRMS)
    microroughness: optics.surface.sag.RoughnessRMS = dataclasses.field(default_factory=optics.surface.sag.RoughnessRMS)
    num_sides: int = 0
    clear_half_width: u.Quantity = 0 * u.mm
    border_width: u.Quantity = 0 * u.mm
    material: optics.surface.material.MultilayerMirror = dataclasses.field(
        default_factory=optics.surface.material.MultilayerMirror)

    @property
    def focal_length(self) -> u.Quantity:
        return self.radius / 2

    @property
    def clear_radius(self) -> u.Quantity:
        return self.clear_half_width / np.cos(360 * u.deg / self.num_sides / 2)

    @property
    def mech_radius(self) -> u.Quantity:
        return (self.clear_half_width + self.border_width) / np.cos(360 * u.deg / self.num_sides / 2)

    @property
    def mech_half_width(self) -> u.Quantity:
        return self.clear_half_width + self.border_width

    @property
    def surface(self) -> SurfaceT:
        surface = super().surface  # type: SurfaceT
        surface.sag = optics.surface.sag.Standard(
            radius=-self.radius,
            conic=self.conic,
        )
        surface.material = self.material.copy()
        surface.aperture = optics.surface.aperture.RegularPolygon(
            radius=self.clear_radius,
            num_sides=self.num_sides,
            # offset_angle=180 * u.deg / self.num_sides,
        )
        surface.aperture_mechanical = optics.surface.aperture.RegularPolygon(
            radius=self.mech_radius,
            num_sides=self.num_sides,
            # offset_angle=180 * u.deg / self.num_sides,
        )
        return surface

    @property
    def dataframe(self) -> pandas:
        dataframe = super().dataframe
        dataframe['radius'] = [format.quantity(self.radius.to(u.mm))]
        dataframe['conic constant'] = [format.quantity(self.conic)]
        dataframe['slope error'] = [format.quantity(self.slope_error.value)]
        dataframe['ripple'] = [format.quantity(self.ripple.value)]
        dataframe['microroughness'] = [format.quantity(self.microroughness.value)]
        dataframe['number of sides'] = [self.num_sides]
        dataframe['clear half-width'] = [format.quantity(self.clear_half_width.to(u.mm))]
        dataframe['border width'] = [format.quantity(self.border_width.to(u.mm))]
        dataframe['substrate thickness'] = [format.quantity(self.substrate_thickness.to(u.mm))]
        return dataframe
