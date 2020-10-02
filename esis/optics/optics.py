import typing as typ
import dataclasses
import timeit
import numpy as np
import scipy.optimize
import astropy.units as u
from kgpy import Name, mixin, vector, optics, transform
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
    def num_channels(self) -> int:
        return self.system.shape[0]

    @property
    def system(self) -> optics.System:
        if self._system is None:
            self._system = optics.System(
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
        return self._system

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

    def _rough_fit_factory(self, x: np.ndarray, channel_index: int, ) -> 'Optics':
        other = self.copy()
        other.grating.roll[channel_index] = x[0] << u.deg
        # other.grating.inclination[channel_index] = x[1] << u.deg
        # other.grating.twist[channel_index] = x[2] << u.deg
        # other.detector.cylindrical_radius[channel_index] = x[0] << u.mm
        # other.detector.cylindrical_azimuth[channel_index] = x[1] << u.deg
        # other.detector.piston[channel_index] = x[3] << u.mm
        # other.detector.inclination[channel_index] = x[1] << u.deg
        # other.detector.roll[channel_index] = x[2] << u.deg
        # other.detector.twist[channel_index] = x[6] << u.deg
        return other

    def _rough_fit_func(
            self, x: np.ndarray, images: u.Quantity, channel_index: int, spatial_samples: int = 100) -> float:

        other = self._rough_fit_factory(x, channel_index=channel_index)
        rays = other.rays_output
        distortion = rays.distortion(polynomial_degree=2)
        wavelength = distortion.wavelength[:, ::2, 0, 0]
        spatial_domain = 1 * u.Quantity([other.system.field_min, other.system.field_max])
        new_images = distortion.distort_cube(
            cube=images,
            wavelength=wavelength,
            spatial_domain_output=spatial_domain,
            spatial_samples_output=spatial_samples,
            inverse=True,
        )
        vignetting = rays.vignetting(polynomial_degree=1)
        new_images = vignetting(
            cube=new_images,
            wavelength=wavelength,
            spatial_domain=spatial_domain,
            inverse=True,
        )

        new_images = new_images / np.nanmean(new_images, axis=(~1, ~0))[..., None, None]

        # ish = new_images.shape[~1:]
        # ix, iy = np.indices(ish)
        # sx, sy = ish[vector.ix] // 2, ish[vector.iy] // 2
        # ix, iy = ix - sx, iy - sy
        # ix, iy = ix / sx, iy / sy
        # ir = np.exp(-np.power(ix * ix + iy * iy, 6))
        # ir = np.exp(-ix * ix / (10 * sx * sx) - iy * iy / (10 * sy * sy))
        # ir = np.sqrt(ix * ix + iy * iy)
        # mask = np.ones_like(ir)
        # mask[ir > sx] = 0
        # norm = -np.nansum(new_images[channel_index])
        norm = np.sqrt(np.nanmean(np.square(new_images[channel_index, 1] - new_images[channel_index, 0])))
        # norm = -np.nansum(ir * new_images[channel_index])
        # norm /= new_images[channel_index].size
        # norm = -np.nansum(new_images[channel_index]) / new_images[channel_index].size
        print('channel index', channel_index)
        print('grating.roll', other.grating.roll[channel_index])
        # print('grating.inclination', other.grating.inclination[channel_index])
        # print('grating.twist', other.grating.twist[channel_index])
        # print('detector.cylindrical_radius', other.detector.cylindrical_radius[channel_index])
        # print('detector.cylindrical azimuth', other.detector.cylindrical_azimuth[channel_index])
        # print('detector.piston', other.detector.piston[channel_index])
        # print('detector.inclination', other.detector.inclination[channel_index])
        # print('detector.roll', other.detector.roll[channel_index])
        # print('detector.twist', other.detector.twist[channel_index])
        print(norm)
        print()
        return norm

    def rough_fit_to_images(self, images: u.Quantity, spatial_samples: int = 100) -> 'Optics':

        images = np.broadcast_to(images[:, None], images.shape[:1] + (2,) + images.shape[1:], subok=True)

        sh = images.shape[:1]

        other = self.copy()

        other.grating.roll = np.broadcast_to(other.grating.roll, sh, subok=True)
        # other.grating.inclination = np.broadcast_to(other.grating.inclination, sh, subok=True)
        # other.grating.twist = np.broadcast_to(other.grating.twist, sh, subok=True)

        # other.detector.cylindrical_radius = np.broadcast_to(other.detector.cylindrical_radius, sh, subok=True)
        # other.detector.cylindrical_azimuth = np.broadcast_to(other.detector.cylindrical_azimuth, sh, subok=True)
        # other.detector.piston = np.broadcast_to(other.detector.piston, sh, subok=True)
        # other.detector.inclination = np.broadcast_to(other.detector.inclination, sh, subok=True)
        # other.detector.roll = np.broadcast_to(other.detector.roll, sh, subok=True)
        # other.detector.twist = np.broadcast_to(other.detector.twist, sh, subok=True)

        for channel_index in range(sh[0]):
            print('channel', channel_index)

            g_roll = other.grating.roll[channel_index]
            # g_inclination = other.grating.inclination[channel_index]
            # g_twist = other.grating.twist[channel_index]
            # d_r = other.detector.cylindrical_radius[channel_index]
            # d_phi = other.detector.cylindrical_azimuth[channel_index]
            # d_z = other.detector.piston[channel_index]
            # d_inclination = other.detector.inclination[channel_index]
            # d_roll = other.detector.roll[channel_index]
            # d_twist = other.detector.twist[channel_index]

            # result = scipy.optimize.differential_evolution(
            result = scipy.optimize.minimize(
                fun=other._rough_fit_func,
                x0=np.array([
                    g_roll.value,
                    # g_inclination.value,
                    # g_twist.value,
                    # d_r.value,
                    # d_phi.value,
                    # d_z.value,
                    # d_inclination.value,
                    # d_roll.value,
                    # d_twist.value,
                ]),
                bounds=np.array([
                    (g_roll[..., None] + [-0.2, 0.2] * u.deg).value,
                    # (g_inclination[..., None] + [-0.1, 0.1] * u.deg).value,
                    # (g_twist[..., None] + [-0.1, 0.1] * u.deg).value,

                    # (d_r[..., None] + [-2, 2] * u.mm).value,
                    # (d_phi[..., None] + [-1, 1] * u.deg).value,
                    # (d_z[..., None] + [-5, 5] * u.mm).value,
                    # (d_inclination[..., None] + [-0.1, 0.1] * u.deg).value,
                    # (d_roll[..., None] + [-1, 1] * u.deg).value,
                    # (d_twist[..., None] + [-.1, .1] * u.deg).value,
                ]),
                method='L-BFGS-B',
                options={
                    'gtol': 1e-2,
                    'eps': 1e-3,
                },
                args=(images.value, channel_index, spatial_samples),
            )

            other = other._rough_fit_factory(result.x, channel_index=channel_index)

        return other

    def _fit_factory(self, x: np.ndarray, ) -> 'Optics':
        other = self.copy()
        other.grating.roll = x[0:4] << u.deg
        other.grating.inclination = x[4:8] << u.deg
        other.grating.twist = x[8:12] << u.deg
        other.grating.piston = x[12:16] << u.mm
        other.grating.cylindrical_radius[16:20]
        # other.detector.cylindrical_radius = x[4:8] << u.mm
        # other.detector.cylindrical_azimuth = x[8:12] << u.deg
        other.detector.piston = x[12:16] << u.mm
        other.detector.inclination = x[16:20] << u.deg
        other.detector.roll = x[20:24] << u.deg
        other.detector.twist = x[24:28] << u.deg
        # other.grating.piston = x[28:32] * u.mm
        # other.wavelengths[~0] = x[32] * u.AA
        other.grating.ruling_density = x[28:32] / u.mm
        # other.grating.tangential_radius = x[32:36] << u.mm
        # other.grating.sagittal_radius = other.grating.tangential_radius
        # other.update()

        # ind = 2
        # other.detector.cylindrical_radius[ind] = self.detector.cylindrical_radius
        # other.detector.cylindrical_azimuth[ind] = self.detector.cylindrical_azimuth[ind]
        # other.detector.piston[ind] = self.detector.piston
        # other.detector.inclination[ind] = self.detector.inclination

        return other

    def _fit_func(self, x: np.ndarray, images: u.Quantity, spatial_samples: int = 100) -> float:

        other = self._fit_factory(x)

        distortion = other.rays_output.distortion(polynomial_degree=2)
        wavelength = distortion.wavelength[:, ::2, 0, 0]
        spatial_domain = 1.1 * u.Quantity([other.system.field_min, other.system.field_max])
        new_images = distortion.distort_cube(
            cube=images,
            wavelength=wavelength,
            spatial_domain_output=spatial_domain,
            spatial_samples_output=spatial_samples,
            inverse=True,
        )

        vignetting = other.rays_output.vignetting(polynomial_degree=1)
        new_images = vignetting(
            cube=new_images,
            wavelength=wavelength,
            spatial_domain=spatial_domain,
            inverse=True,
        )

        # norm = -np.power(np.nanmean(new_images.prod((0, 1))), 1 / (new_images.shape[0] * new_images.shape[1]))

        new_images = new_images / np.nanmean(new_images, axis=(~1, ~0))[..., None, None]
        # norm = np.sqrt(np.nanmean(np.square(new_images[3] - new_images[2])))
        # norm += np.sqrt(np.nanmean(np.square(new_images[1] - new_images[2])))
        # norm += np.sqrt(np.nanmean(np.square(new_images[0] - new_images[2])))
        # norm += np.sqrt(np.nanmean(np.square(new_images[:, ~0] - new_images[:, 0])))
        norm = np.sqrt(np.nanmean(np.square(np.roll(new_images, 1, axis=0) - new_images)))
        norm += np.sqrt(np.nanmean(np.square(np.roll(new_images, 2, axis=0) - new_images)))
        norm += np.sqrt(np.nanmean(np.square(np.roll(new_images, 3, axis=0) - new_images)))
        norm += np.sqrt(np.nanmean(np.square(np.roll(new_images, 1, axis=0) - np.roll(new_images, 1, axis=1))))
        norm += np.sqrt(np.nanmean(np.square(np.roll(new_images, 2, axis=0) - np.roll(new_images, 1, axis=1))))
        norm += np.sqrt(np.nanmean(np.square(np.roll(new_images, 3, axis=0) - np.roll(new_images, 1, axis=1))))
        norm /= 6

        # print('grating.piston', other.grating.piston)
        print('grating.roll', other.grating.roll)
        print('grating.inclination', other.grating.inclination)
        print('grating.twist', other.grating.twist)
        # print('detector.cylindrical_radius', other.detector.cylindrical_radius)
        # print('detector.cylindrical azimuth', other.detector.cylindrical_azimuth)
        print('detector.piston', other.detector.piston)
        print('detector.inclination', other.detector.inclination)
        print('detector.roll', other.detector.roll)
        print('detector.twist', other.detector.twist)
        # print('OV wavelength', other.wavelengths[~0])
        # print('grating.ruling_density', other.grating.ruling_density)
        # print('grating.radius', other.grating.tangential_radius)
        # print('grating.ruling_density', other.grating.ruling_density)
        print(norm)
        print()
        return norm

    def fit_to_images(self, images: u.Quantity, spatial_samples: int = 100) -> 'Optics':

        images = np.broadcast_to(images[:, None], images.shape[:1] + (2,) + images.shape[1:], subok=True)

        sh = images.shape[:1]

        other = self.copy()

        g_roll = np.broadcast_to(other.grating.roll, sh, subok=True)
        g_inclination = np.broadcast_to(other.grating.inclination, sh, subok=True)
        g_twist = np.broadcast_to(other.grating.twist, sh, subok=True)

        d_r = np.broadcast_to(other.detector.cylindrical_radius, sh, subok=True)
        d_phi = np.broadcast_to(other.detector.cylindrical_azimuth, sh, subok=True)
        d_z = np.broadcast_to(other.detector.piston, sh, subok=True)
        d_inclination = np.broadcast_to(other.detector.inclination, sh, subok=True)
        d_roll = np.broadcast_to(other.detector.roll, sh, subok=True)
        d_twist = np.broadcast_to(other.detector.twist, sh, subok=True)
        # g_z = np.broadcast_to(other.grating.piston, sh, subok=True)
        g_T = np.broadcast_to(other.grating.ruling_density, sh, subok=True)
        g_r = np.broadcast_to(other.grating.tangential_radius, sh, subok=True)
        # wavl = other.wavelengths[~0:]

        # result = scipy.optimize.minimize(
        result = scipy.optimize.differential_evolution(
            # fun=other._fit_func,
            func=other._fit_func,
            # x0=np.concatenate([
            #     g_roll.value,
            #     # g_inclination.value,
            #     # g_twist.value,
            #     d_r.value,
            #     d_phi.value,
            #     d_z.value,
            #     d_inclination.value,
            #     d_roll.value,
            #     d_twist.value,
            #     # g_z.value,
            #     # g_T.value,
            #     # g_r.value,
            #     # wavl.value,
            # ]),
            bounds=np.concatenate([
                (g_roll[..., None] + [-2, 2] * u.deg).value,
                (g_inclination[..., None] + [-0.02, 0.02] * u.deg).value,
                (g_twist[..., None] + [-0.02, 0.02] * u.deg).value,
                # (d_r[..., None] + [-1, 1] * u.mm).value,
                # (d_phi[..., None] + [-2, 2] * u.deg).value,
                (d_z[..., None] + [-5, 5] * u.mm).value,
                (d_inclination[..., None] + [-0.5, 0.5] * u.deg).value,
                (d_roll[..., None] + [-2, 2] * u.deg).value ,
                (d_twist[..., None] + [-0.2, 0.2] * u.deg).value,
                # (g_z[..., None] + [-0.1, 0.1] * u.mm).value,
                (g_T[..., None] + [-4, 4] / u.mm).value,

                # (g_r[..., None] + [-5, 5] * u.mm).value,
                # (wavl[..., None] + [-1, 1] * u.AA).value,
            ]),
            # method='L-BFGS-B',
            # jac='3-point',
            # options={
                # 'gtol': 1e-2,
                # 'eps': 5e-4,
                # 'maxcor': 1000,
                # 'finite_diff_rel_step': 1e-3,
            # },
            args=(images.value, spatial_samples),
            # T=1.,
            # stepsize=0.1,
            disp=True,
            mutation=(0.05, 0.5),
            popsize=5,
            # minimizer_kwargs={
            #     'method': 'L-BFGS-B',
            #     'args': (images.value, spatial_samples),
            #     'options': {
            #         'gtol': 1e-2,
            #         'eps': 5e-4,
            #         'maxcor': 1000,
            #     }
            # },
            # callback=lambda x, f, context: print('context', context),
            # local_search_options={
            #     'options': {'eps': 1e-4, 'ftol': 1e-3},
            # }
        )

        return other._fit_factory(result.x)

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
