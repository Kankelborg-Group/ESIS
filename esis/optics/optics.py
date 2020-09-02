import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
from kgpy import Name, mixin, vector, optics
from . import components as comps

__all__ = ['Optics']


@dataclasses.dataclass
class Optics(mixin.Named):
    """
    Add test docstring to see if this is the problem.
    """
    name: Name = dataclasses.field(default_factory=lambda: Name('ESIS'))
    components: comps.Components = dataclasses.field(default_factory=lambda: comps.Components())
    wavelengths: u.Quantity = 0 * u.nm
    pupil_samples: int = 10
    field_samples: int = 10

    def __post_init__(self):
        self.update()

    def update(self) -> typ.NoReturn:
        self._system = None

    @property
    def system(self) -> optics.System:
        if self._system is None:
            self._system = self._calc_system()
        return self._system

    def _calc_system(self) -> optics.System:
        return optics.System(
            object_surface=self.components.source.surface,
            surfaces=optics.SurfaceList(self.components),
            wavelengths=self.wavelengths,
            pupil_samples=self.pupil_samples,
            field_samples=self.field_samples,
        )

    @property
    def back_focal_length(self) -> u.Quantity:
        return -self.components.detector.piston

    @property
    def magnification(self) -> u.Quantity:
        grating = self.components.grating
        detector = self.components.detector
        source_pos = vector.from_components(self.components.primary.focal_length)
        grating_pos = vector.from_components(grating.piston, grating.cylindrical_radius)
        detector_pos = vector.from_components(detector.piston, detector.cylindrical_radius)
        entrance_arm = grating_pos - source_pos
        exit_arm = detector_pos - grating_pos
        return vector.length(exit_arm, keepdims=False) / vector.length(entrance_arm, keepdims=False)

    @property
    def effective_focal_length(self) -> u.Quantity:
        return self.magnification * self.components.primary.focal_length

    @property
    def pixel_subtent(self):
        return np.arctan2(self.components.detector.pixel_width, self.effective_focal_length) << u.rad

    def copy(self) -> 'Optics':
        other = super().copy()  # type: Optics
        other.components = self.components.copy()
        other.wavelengths = self.wavelengths.copy()
        other.pupil_samples = self.pupil_samples
        other.field_samples = self.field_samples
        return other

    def apply_poletto_layout(
            self,
            wavelength_1: u.Quantity,
            wavelength_2: u.Quantity,
            magnification: u.Quantity,
            obscuration_margin: u.Quantity,
            obscuration_thickness: u.Quantity,
            image_margin: u.Quantity,
            detector_is_opposite_grating: bool = False,
            use_toroidal_grating: bool = False,
            use_vls_grating: bool = False,
    ) -> 'Optics':
        other = self.copy()

        c = other.components

        num_sides = c.primary.num_sides
        wedge_half_angle = c.primary.surface.aperture.half_edge_subtent

        primary_clear_radius = c.primary.surface.aperture.min_radius
        detector_half_width = -c.detector.surface.aperture_mechanical.width_x_neg + c.detector.dynamic_clearance
        c.detector.cylindrical_radius = primary_clear_radius + detector_half_width
        if detector_is_opposite_grating:
            c.detector.cylindrical_radius = -c.detector.cylindrical_radius

        c.grating = c.grating.apply_gregorian_layout(
            magnification=magnification,
            primary_focal_length=c.primary.focal_length,
            primary_clear_radius=c.primary.clear_radius,
            back_focal_length=other.back_focal_length,
            detector_cylindrical_radius=c.detector.cylindrical_radius,
            obscuration_margin=obscuration_margin,
        )
        c.grating = c.grating.apply_poletto_prescription(
            wavelength_1=wavelength_1,
            wavelength_2=wavelength_2,
            magnification=magnification,
            primary_focal_length=c.primary.focal_length,
            detector_cylindrical_radius=c.detector.cylindrical_radius,
            is_toroidal=use_toroidal_grating,
            is_vls=use_vls_grating,
        )
        c.grating.aper_half_angle = c.primary.surface.aperture.half_edge_subtent

        c.detector = c.detector.apply_poletto_prescription(
            wavelength_1=wavelength_1,
            wavelength_2=wavelength_2,
            magnification=magnification,
            primary_focal_length=c.primary.focal_length,
            grating=c.grating,
        )

        detector_half_height = c.detector.surface.aperture.half_width_y
        undersize_factor = (detector_half_height - image_margin) / detector_half_height
        fov_min_radius = other.pixel_subtent * undersize_factor * c.detector.num_pixels[vector.iy] / 2
        pixel_klooge = 4
        fs_half_radius = fov_min_radius + pixel_klooge * other.pixel_subtent
        c.field_stop.clear_radius = c.primary.focal_length * np.tan(fs_half_radius) / np.cos(wedge_half_angle)
        c.field_stop.piston = c.primary.focal_length
        c.field_stop.num_sides = num_sides

        c.source.half_width_x = fov_min_radius.to(u.arcmin)
        c.source.half_width_y = fov_min_radius.to(u.arcmin)

        output_angle = c.grating.inclination + c.grating.nominal_output_angle
        piston_fg = (c.filter.piston - c.grating.piston)
        c.filter.cylindrical_radius = c.grating.cylindrical_radius - piston_fg * np.tan(output_angle)
        c.filter.inclination = -output_angle

        c.central_obscuration.piston = c.grating.piston + obscuration_thickness
        c.central_obscuration.obscured_radius = c.grating.outer_clear_radius + obscuration_margin
        c.central_obscuration.num_sides = num_sides

        c.front_aperture.piston = c.central_obscuration.piston + 100 * u.mm
        # c.front_aperture.clear_radius = c.detector.channel_radius + c.detector.main_surface.aperture.width_x_pos

        other.wavelengths = u.Quantity([wavelength_1, (wavelength_1 + wavelength_2) / 2, wavelength_2])

        other.update()

        return other

