import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
from kgpy import Name, mixin, vector, optics
from . import Source, FrontAperture, CentralObscuration, Primary, FieldStop, Grating, Filter, Detector

__all__ = ['Optics']


@dataclasses.dataclass
class Optics(mixin.Named):
    """
    Add test docstring to see if this is the problem.
    """
    name: Name = dataclasses.field(default_factory=lambda: Name('ESIS'))
    wavelengths: u.Quantity = 0 * u.nm
    pupil_samples: int = 10
    field_samples: int = 10
    source: Source = dataclasses.field(default_factory=Source)
    front_aperture: FrontAperture = dataclasses.field(default_factory=FrontAperture)
    central_obscuration: CentralObscuration = dataclasses.field(default_factory=CentralObscuration)
    primary: Primary = dataclasses.field(default_factory=Primary)
    field_stop: FieldStop = dataclasses.field(default_factory=FieldStop)
    grating: Grating = dataclasses.field(default_factory=Grating)
    filter: Filter = dataclasses.field(default_factory=Filter)
    detector: Detector = dataclasses.field(default_factory=Detector)

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
            object_surface=self.source.surface,
            surfaces=optics.SurfaceList([
                self.front_aperture.surface,
                self.central_obscuration.surface,
                self.central_obscuration.surface,
                self.primary.surface,
                self.field_stop.surface,
                self.grating.surface,
                self.filter.surface,
                self.detector.surface,
            ]),
            wavelengths=self.wavelengths,
            pupil_samples=self.pupil_samples,
            field_samples=self.field_samples,
        )

    @property
    def rays_output(self) -> optics.Rays:
        rays = self.system.rays_output.copy()
        rays.position = rays.position / self.detector.pixel_width.to(u.mm) * u.pix
        rays.position[vector.x] = rays.position[vector.x] + self.detector.num_pixels[vector.ix] * u.pix / 2
        rays.position[vector.y] = rays.position[vector.y] + self.detector.num_pixels[vector.iy] * u.pix / 2
        return rays

    @property
    def back_focal_length(self) -> u.Quantity:
        return -self.detector.piston

    @property
    def magnification(self) -> u.Quantity:
        grating = self.grating
        detector = self.detector
        source_pos = vector.from_components(self.primary.focal_length)
        grating_pos = vector.from_components(grating.piston, grating.cylindrical_radius)
        detector_pos = vector.from_components(detector.piston, detector.cylindrical_radius)
        entrance_arm = grating_pos - source_pos
        exit_arm = detector_pos - grating_pos
        return vector.length(exit_arm, keepdims=False) / vector.length(entrance_arm, keepdims=False)

    @property
    def effective_focal_length(self) -> u.Quantity:
        return self.magnification * self.primary.focal_length

    @property
    def pixel_subtent(self):
        return np.arctan2(self.detector.pixel_width, self.effective_focal_length) << u.rad

    def copy(self) -> 'Optics':
        other = super().copy()  # type: Optics
        other.wavelengths = self.wavelengths.copy()
        other.pupil_samples = self.pupil_samples
        other.field_samples = self.field_samples
        other.source = self.source.copy()
        other.front_aperture = self.front_aperture.copy()
        other.central_obscuration = self.central_obscuration.copy()
        other.primary = self.primary.copy()
        other.field_stop = self.field_stop.copy()
        other.grating = self.grating.copy()
        other.filter = self.filter.copy()
        other.detector = self.detector.copy()
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

        num_sides = self.primary.num_sides
        wedge_half_angle = self.primary.surface.aperture.half_edge_subtent

        primary_clear_radius = self.primary.surface.aperture.min_radius
        detector_half_width = -self.detector.surface.aperture_mechanical.width_x_neg + self.detector.dynamic_clearance
        self.detector.cylindrical_radius = primary_clear_radius + detector_half_width
        if detector_is_opposite_grating:
            self.detector.cylindrical_radius = -self.detector.cylindrical_radius
            self.grating.diffraction_order = -self.grating.diffraction_order

        self.grating = self.grating.apply_gregorian_layout(
            magnification=magnification,
            primary_focal_length=self.primary.focal_length,
            primary_clear_radius=self.primary.clear_radius,
            back_focal_length=other.back_focal_length,
            detector_cylindrical_radius=self.detector.cylindrical_radius,
            obscuration_margin=obscuration_margin,
        )
        self.grating = self.grating.apply_poletto_prescription(
            wavelength_1=wavelength_1,
            wavelength_2=wavelength_2,
            magnification=magnification,
            primary_focal_length=self.primary.focal_length,
            detector_cylindrical_radius=self.detector.cylindrical_radius,
            is_toroidal=use_toroidal_grating,
            is_vls=use_vls_grating,
        )
        self.grating.aper_half_angle = self.primary.surface.aperture.half_edge_subtent

        self.detector = self.detector.apply_poletto_prescription(
            wavelength_1=wavelength_1,
            wavelength_2=wavelength_2,
            magnification=magnification,
            primary_focal_length=self.primary.focal_length,
            grating=self.grating,
        )

        detector_quarter_width = self.detector.surface.aperture.half_width_x / 2
        undersize_factor = (detector_quarter_width - image_margin) / detector_quarter_width
        fov_min_radius = other.pixel_subtent * undersize_factor * self.detector.num_pixels[vector.ix] / 4
        pixel_klooge = 8
        fs_half_radius = fov_min_radius + pixel_klooge * other.pixel_subtent
        self.field_stop.clear_radius = self.primary.focal_length * np.tan(fs_half_radius) / np.cos(wedge_half_angle)
        self.field_stop.piston = self.primary.focal_length
        self.field_stop.num_sides = num_sides

        self.source.half_width_x = fov_min_radius.to(u.arcmin)
        self.source.half_width_y = fov_min_radius.to(u.arcmin)

        output_angle = self.grating.inclination + self.grating.nominal_output_angle
        self.filter.piston = self.detector.piston + 200 * u.mm
        piston_fg = (self.filter.piston - self.grating.piston)
        self.filter.cylindrical_radius = self.grating.cylindrical_radius - piston_fg * np.tan(output_angle)
        self.filter.inclination = -output_angle

        self.central_obscuration.piston = self.grating.piston + obscuration_thickness
        grating_outer_radius = self.grating.cylindrical_radius + self.grating.outer_half_width
        self.central_obscuration.obscured_half_width = grating_outer_radius + obscuration_margin
        self.central_obscuration.num_sides = num_sides

        self.front_aperture.piston = self.central_obscuration.piston + 100 * u.mm
        # self.front_aperture.clear_radius = self.detector.channel_radius + self.detector.main_surface.aperture.width_x_pos

        other.wavelengths = u.Quantity([wavelength_1, (wavelength_1 + wavelength_2) / 2, wavelength_2])
        # other.wavelengths = u.Quantity([wavelength_1, wavelength_2])

        other.update()

        return other

