import typing as typ
import dataclasses
import pathlib
import timeit
import numpy as np
import scipy.optimize
import scipy.signal
import matplotlib.transforms
import matplotlib.axes
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import astropy.time
import kgpy.transform
from kgpy import Name, mixin, vector, optics, observatories, polynomial, grid, plot
from . import Source, FrontAperture, CentralObscuration, Primary, FieldStop, Grating, Filter, Detector

__all__ = ['Optics']


@dataclasses.dataclass
class Optics(
    mixin.Pickleable,
    mixin.Named,
):
    """
    Add test docstring to see if this is the problem.
    """
    name: Name = dataclasses.field(default_factory=lambda: Name('ESIS'))
    source: Source = dataclasses.field(default_factory=Source)
    front_aperture: FrontAperture = dataclasses.field(default_factory=FrontAperture)
    central_obscuration: typ.Optional[CentralObscuration] = dataclasses.field(default_factory=CentralObscuration)
    primary: Primary = dataclasses.field(default_factory=Primary)
    field_stop: FieldStop = dataclasses.field(default_factory=FieldStop)
    grating: Grating = dataclasses.field(default_factory=Grating)
    filter: typ.Optional[Filter] = dataclasses.field(default_factory=Filter)
    detector: Detector = dataclasses.field(default_factory=Detector)
    wavelength: u.Quantity = 0 * u.nm
    field_samples: typ.Union[int, vector.Vector2D] = 10
    field_is_stratified_random: bool = False
    pupil_samples: typ.Union[int, vector.Vector2D] = 10
    pupil_is_stratified_random: bool = False
    grid_velocity_los: grid.Grid1D = dataclasses.field(default_factory=lambda: grid.RegularGrid1D(
        min=0 * u.km / u.s,
        max=0 * u.km / u.s,
        num_samples=1,
    ))
    pointing: vector.Vector2D = dataclasses.field(default_factory=vector.Vector2D.angular)
    roll: u.Quantity = 0 * u.deg
    stray_light: u.Quantity = 0 * u.adu
    vignetting_correction: polynomial.Polynomial3D = dataclasses.field(
        default_factory=lambda: polynomial.Polynomial3D(
            degree=1,
            coefficients=[1 * u.dimensionless_unscaled, 0 / u.AA, 0 / u.arcsec, 0 / u.arcsec]
        )
    )

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
            surfaces = optics.surface.SurfaceList()
            surfaces.append(self.front_aperture.surface)
            if self.central_obscuration is not None:
                surfaces.append(self.central_obscuration.surface)
            surfaces.append(self.primary.surface)
            surfaces.append(self.field_stop.surface)
            surfaces.append(self.grating.surface)
            if self.filter is not None:
                surfaces.append(self.filter.surface)
            surfaces.append(self.detector.surface)
            self._system = optics.System(
                object_surface=self.source.surface,
                surfaces=surfaces,
                wavelength=self.wavelength,
                field_samples=self.field_samples,
                field_is_stratified_random=self.field_is_stratified_random,
                pupil_samples=self.pupil_samples,
                pupil_is_stratified_random=self.pupil_is_stratified_random,
                grid_velocity_los=self.grid_velocity_los,
                pointing=self.pointing,
                roll=self.roll,
            )
        return self._system

    @property
    def rays_output(self) -> optics.rays.Rays:
        rays = self.system.rays_output.copy()
        rays.position = rays.position / (self.detector.pixel_width.to(u.mm) / u.pix)
        rays.position.x = rays.position.x + self.detector.num_pixels[vector.ix] * u.pix / 2
        rays.position.y = rays.position.y + self.detector.num_pixels[vector.iy] * u.pix / 2
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
    def pixel_subtent(self) -> u.Quantity:
        return np.arctan2(self.detector.pixel_width, self.effective_focal_length) << u.rad

    @property
    def plate_scale(self) -> vector.Vector2D:
        return self.system.rays_output.distortion().plate_scale[0].max() * (self.detector.pixel_width.to(u.mm) / u.pix)

    @property
    def dispersion(self) -> u.Quantity:
        val = self.system.rays_output.distortion().dispersion.max() * (self.detector.pixel_width.to(u.mm) / u.pix)
        return val.to(u.Angstrom / u.pix)

    def copy(self) -> 'Optics':
        other = super().copy()  # type: Optics
        other.wavelength = self.wavelength.copy()
        other.pupil_samples = self.pupil_samples
        other.field_samples = self.field_samples
        other.source = self.source.copy()
        other.front_aperture = self.front_aperture.copy()
        if self.central_obscuration is not None:
            other.central_obscuration = self.central_obscuration.copy()
        else:
            other.central_obscuration = self.central_obscuration
        other.primary = self.primary.copy()
        other.field_stop = self.field_stop.copy()
        other.grating = self.grating.copy()
        if self.filter is not None:
            other.filter = self.filter.copy()
        else:
            other.filter = self.filter
        other.detector = self.detector.copy()
        other.pointing = self.pointing.copy()
        other.roll = self.roll.copy()
        other.stray_light = self.stray_light.copy()
        other.vignetting_correction = self.vignetting_correction.copy()
        return other

    def __call__(
            self,
            data: u.Quantity,
            wavelength: u.Quantity,
            spatial_domain_input: u.Quantity,
            spatial_domain_output: u.Quantity,
            spatial_samples_output: typ.Union[int, typ.Tuple[int, int]],
            inverse: bool = False,
    ):
        distortion = self.rays_output.distortion(polynomial_degree=2)
        vignetting = self.rays_output.vignetting(polynomial_degree=1)
        if not inverse:
            data = vignetting(
                cube=data,
                wavelength=wavelength,
                spatial_domain=spatial_domain_input,
                inverse=inverse,
            )
            data = distortion.distort_cube(
                cube=data,
                wavelength=wavelength,
                spatial_domain_input=spatial_domain_input,
                spatial_domain_output=spatial_domain_output,
                spatial_samples_output=spatial_samples_output,
                inverse=inverse,
                fill_value=np.nan,
            )
        else:
            data = distortion.distort_cube(
                cube=data,
                wavelength=wavelength,
                spatial_domain_input=spatial_domain_input,
                spatial_domain_output=spatial_domain_output,
                spatial_samples_output=spatial_samples_output,
                inverse=inverse,
                fill_value=np.nan,
            )
            data = vignetting(
                cube=data,
                wavelength=wavelength,
                spatial_domain=spatial_domain_output,
                inverse=inverse,
            )
        return data

    def _rough_fit_factory_simple(self, x: np.ndarray, channel_index: int) -> 'Optics':
        other = self.copy()
        (
            g_twist,
            g_inclination,
            g_roll,
            # d_roll,
            g_T,
            # d_inclination,
            # d_twist,
            # d_piston,
        ) = x
        other.grating.inclination[channel_index] = g_inclination << u.deg
        other.grating.twist[channel_index] = g_twist << u.deg
        other.grating.roll[channel_index] = g_roll << u.deg
        # other.detector.roll[channel_index] = d_roll << u.deg
        other.grating.ruling_density[channel_index] = g_T / u.mm
        # other.detector.inclination[channel_index] = d_inclination << u.deg
        # other.detector.twist[channel_index] = d_twist << u.deg
        # other.detector.piston[channel_index] = d_piston << u.mm
        return other

    def _rough_fit_factory(self, x: np.ndarray, channel_index: int) -> 'Optics':
        other = self.copy()
        (
            g_twist,
            g_inclination,
            g_roll,
            d_roll,
            g_T,
            d_inclination,
            d_twist,
            d_piston,
            # g_piston,
            v_0,
            v_x,
            v_y,
            stray_light,
        ) = x
        other.grating.inclination[channel_index] = g_inclination << u.deg
        other.grating.twist[channel_index] = g_twist << u.deg
        other.grating.roll[channel_index] = g_roll << u.deg
        other.detector.roll[channel_index] = d_roll << u.deg
        other.grating.ruling_density[channel_index] = g_T / u.mm
        other.detector.inclination[channel_index] = d_inclination << u.deg
        other.detector.twist[channel_index] = d_twist << u.deg
        other.detector.piston[channel_index] = d_piston << u.mm
        # other.grating.piston[channel_index] = g_piston << u.mm
        other.vignetting_correction.coefficients[0][channel_index] = v_0 * u.dimensionless_unscaled
        other.vignetting_correction.coefficients[2][channel_index] = v_x / u.arcsec
        other.vignetting_correction.coefficients[3][channel_index] = v_y / u.arcsec
        other.stray_light[channel_index] = stray_light * u.adu

        return other

    def _rough_fit_func(
            self,
            x: np.ndarray,
            images: u.Quantity,
            channel_index: int,
            ref_image: u.Quantity = None,
            spatial_samples: int = 100,
            oversize_ratio: float = 1.1,
            is_simple: bool = False,
            plot_steps: bool = False,
    ) -> float:

        if is_simple:
            other = self._rough_fit_factory_simple(x, channel_index=channel_index)
        else:
            other = self._rough_fit_factory(x, channel_index=channel_index)

        other.grating.roll = other.grating.roll[channel_index]
        other.grating.inclination = other.grating.inclination[channel_index]
        other.grating.twist = other.grating.twist[channel_index]
        other.detector.cylindrical_radius = other.detector.cylindrical_radius[channel_index]
        other.detector.cylindrical_azimuth = other.detector.cylindrical_azimuth[channel_index]
        other.detector.piston = other.detector.piston[channel_index]
        other.detector.roll = other.detector.roll[channel_index]
        other.detector.inclination = other.detector.inclination[channel_index]
        other.detector.twist = other.detector.twist[channel_index]
        other.grating.ruling_density = other.grating.ruling_density[channel_index]
        other.grating.cylindrical_azimuth = other.grating.cylindrical_azimuth[channel_index]
        other.grating.piston = other.grating.piston[channel_index]
        other.filter.cylindrical_azimuth = other.filter.cylindrical_azimuth[channel_index]
        other.stray_light = other.stray_light[channel_index]
        other.vignetting_correction.coefficients[0] = other.vignetting_correction.coefficients[0][channel_index]
        other.vignetting_correction.coefficients[1] = other.vignetting_correction.coefficients[1][channel_index]
        other.vignetting_correction.coefficients[2] = other.vignetting_correction.coefficients[2][channel_index]
        other.vignetting_correction.coefficients[3] = other.vignetting_correction.coefficients[3][channel_index]

        ish = 2 * (spatial_samples, )
        ix, iy = np.indices(ish)
        sx, sy = ish[vector.ix] // 2, ish[vector.iy] // 2
        ix, iy = ix - sx, iy - sy
        sx, sy = sx / oversize_ratio, sy / oversize_ratio
        sr = np.sqrt(sx * sx + sy * sy)
        ir = np.ones_like(ix, dtype=np.float)
        ir[iy > ix + sr] = 0
        ir[iy < ix - sr] = 0
        ir[iy > -ix + sr] = 0
        ir[iy < -ix - sr] = 0
        ir[ix > sx] = 0
        ir[ix < -sx] = 0
        ir[iy > sy] = 0
        ir[iy < -sy] = 0

        # mask_584 = ir * u.adu
        # mask_584[ir > 0] = np.nan
        # mask_584 = np.broadcast_to(mask_584, (2,) + mask_584.shape, subok=True)

        rays = other.rays_output
        distortion = rays.distortion(polynomial_degree=2)
        wavelength = distortion.wavelength[..., ::2, 0, 0]
        spatial_domain = oversize_ratio * u.Quantity([other.system.field_min[:2], other.system.field_max[:2]])
        pixel_domain = ([[0, 0], images.shape[~1:]] * u.pix)
        # w584 = distortion.wavelength[..., 1, 0, 0]
        # w584 = np.broadcast_to(w584[..., None], w584.shape + (2, ), subok=True)
        new_images = images[channel_index] - other.stray_light
        # new_images += distortion.distort_cube(
        #     cube=mask_584,
        #     wavelength=w584,
        #     spatial_domain_input=spatial_domain,
        #     spatial_domain_output=pixel_domain,
        #     spatial_samples_output=images.shape[~1:],
        # )

        new_images = distortion.distort_cube(
            cube=new_images,
            wavelength=wavelength,
            spatial_domain_input=pixel_domain,
            spatial_domain_output=spatial_domain,
            spatial_samples_output=spatial_samples,
            inverse=True,
            fill_value=np.nan,
        )
        vignetting = rays.vignetting(polynomial_degree=1)
        new_images = vignetting(
            cube=new_images,
            wavelength=wavelength,
            spatial_domain=spatial_domain,
            inverse=True,
        )
        new_images = optics.aberration.Vignetting.apply_model(
            model=other.vignetting_correction,
            cube=new_images,
            wavelength=wavelength,
            spatial_domain=spatial_domain
        )


        # aia_images = aia_obs.intensity[0, 0]
        # aia_images = aia_images / aia_images.mean()
        # new_images = new_images / np.nanmean(new_images, axis=(~1, ~0))[..., None, None]
        # new_images = new_images - np.nanmean(new_images, axis=(~1, ~0))[..., None, None]

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


        ir = np.broadcast_to(ir, new_images.shape)
        ir = ir / np.nanmean(ir)
        if ref_image is None:
            ref_image = ir * np.nanmean(new_images, axis=(~1, ~0))[..., None, None]
        # else:
        #     new_images[ir == 0] = np.nan
        #     ref_image[ir == 0] = np.nan
        #
        # new_images = new_images / np.nanmean(new_images, axis=(~1, ~0))[..., None, None]
        # ref_image = ref_image / np.nanmean(ref_image, axis=(~1, ~0))[..., None, None]



        # norm = np.sqrt(np.nanmean(np.square(new_images - ir)))
        # norm = np.sqrt(np.nanmean(np.square(new_images[channel_index] - ir)))
        norm_base = new_images - ref_image
        # norm_base = scipy.signal.correlate(new_images, ir, mode='same')


        if plot_steps:
            test_img = norm_base.value
            _, axs = plt.subplots(ncols=2, figsize=(12, 6))
            axs[0].imshow(test_img[0], vmin=np.nanpercentile(test_img[0], 2), vmax=np.nanpercentile(test_img[0], 98))
            axs[1].imshow(test_img[1], vmin=np.nanpercentile(test_img[1], 2), vmax=np.nanpercentile(test_img[1], 98))
            plt.show()

        norm = np.sqrt(np.nanmean(np.square(norm_base)))
        # norm = -np.sqrt(np.nanmean(np.square(norm_base.max((~1, ~0)))))
        # norm = -np.nansum(ir * new_images[channel_index])
        # norm /= new_images[channel_index].size
        # norm = -np.nansum(new_images[channel_index] * ir) / new_images[channel_index].size
        # print('channel index', channel_index)
        # print('detector.cylindrical_radius', other.detector.cylindrical_radius)
        # print('detector.cylindrical azimuth', other.detector.cylindrical_azimuth)
        print('grating.inclination', other.grating.inclination)
        print('grating.twist', other.grating.twist)
        print('grating.roll', other.grating.roll)
        # print('detector.roll', other.detector.roll)
        print('grating.ruling_density', other.grating.ruling_density)
        # print('detector.inclination', other.detector.inclination)
        # print('detector.twist', other.detector.twist)
        # print('detector.piston', other.detector.piston)
        # # print('grating.piston', other.grating.piston)
        # print('stray_light', other.stray_light)
        # print('vignetting', other.vignetting_correction.coefficients)
        print(norm)
        # print()

        return norm.value

    def rough_fit_to_images(
            self,
            images: u.Quantity,
            spatial_samples: int = 100,
            ref_index_start: int = 0,
            ref_index_end: int = 2,
            plot_steps: bool = False,
            # aia_obs: observatories.aia.AIA,
    ) -> 'Optics':

        images = np.broadcast_to(images[:, None], images.shape[:1] + (2,) + images.shape[1:], subok=True)

        sh = images.shape[:1]

        other = self.copy()

        other.grating.roll = np.broadcast_to(other.grating.roll, sh, subok=True)
        other.grating.inclination = np.broadcast_to(other.grating.inclination, sh, subok=True)
        other.grating.twist = np.broadcast_to(other.grating.twist, sh, subok=True)

        other.detector.cylindrical_radius = np.broadcast_to(other.detector.cylindrical_radius, sh, subok=True)
        other.detector.cylindrical_azimuth = np.broadcast_to(other.detector.cylindrical_azimuth, sh, subok=True)
        other.detector.piston = np.broadcast_to(other.detector.piston, sh, subok=True)
        other.detector.inclination = np.broadcast_to(other.detector.inclination, sh, subok=True)
        other.detector.roll = np.broadcast_to(other.detector.roll, sh, subok=True)
        other.detector.twist = np.broadcast_to(other.detector.twist, sh, subok=True)
        other.grating.ruling_density = np.broadcast_to(other.grating.ruling_density, sh, subok=True)
        other.grating.piston = np.broadcast_to(other.grating.piston, sh, subok=True)
        other.stray_light = np.broadcast_to(other.stray_light, sh, subok=True)

        v_c = other.vignetting_correction.coefficients
        v_c[0] = np.broadcast_to(v_c[0], sh, subok=True)
        v_c[1] = np.broadcast_to(v_c[1], sh, subok=True)
        v_c[2] = np.broadcast_to(v_c[2], sh, subok=True)
        v_c[3] = np.broadcast_to(v_c[3], sh, subok=True)

        oversize_ratio = 1.1

        ish = 2 * (1024, )
        ix, iy = np.indices(ish)
        sx, sy = ish[vector.ix] // 2, ish[vector.iy] // 2
        ix, iy = ix - sx, iy - sy
        sx, sy = sx / oversize_ratio, sy / oversize_ratio
        sr = np.sqrt(sx * sx + sy * sy)
        ir = np.ones_like(ix, dtype=np.float)
        ir[iy > ix + sr] = 0
        ir[iy < ix - sr] = 0
        ir[iy > -ix + sr] = 0
        ir[iy < -ix - sr] = 0
        ir[ix > sx] = 0
        ir[ix < -sx] = 0
        ir[iy > sy] = 0
        ir[iy < -sy] = 0

        mask_584 = ir * u.adu
        mask_584[ir > 0] = np.nan
        mask_584 = np.broadcast_to(mask_584, (4, 2,) + mask_584.shape, subok=True)

        for ref_index in range(ref_index_start, ref_index_end):
            for channel_index in range(sh[0]):
                print('reference', ref_index)
                print('channel', channel_index)

                g_roll = other.grating.roll[channel_index]
                g_inclination = other.grating.inclination[channel_index]
                g_twist = other.grating.twist[channel_index]
                d_r = other.detector.cylindrical_radius[channel_index]
                d_phi = other.detector.cylindrical_azimuth[channel_index]
                d_z = other.detector.piston[channel_index]
                d_inclination = other.detector.inclination[channel_index]
                d_roll = other.detector.roll[channel_index]
                d_twist = other.detector.twist[channel_index]
                g_T = other.grating.ruling_density[channel_index]
                g_piston = other.grating.piston[channel_index]
                stray_light = other.stray_light[channel_index]
                v_0 = other.vignetting_correction.coefficients[0][channel_index]
                v_x = other.vignetting_correction.coefficients[2][channel_index]
                v_y = other.vignetting_correction.coefficients[3][channel_index]



                if ref_index == 0:
                    is_simple = True
                    x0 = np.array([
                        g_twist.value,
                        g_inclination.value,
                        g_roll.value,
                        # d_roll.value,
                        g_T.value,
                        # d_inclination.value,
                        # d_twist.value,
                        # d_z.value,
                    ])
                    bounds = np.array([
                        (g_twist[..., None] + [-0.05, 0.05] * u.deg).value,
                        (g_inclination[..., None] + [-0.04, 0.04] * u.deg).value,
                        (g_roll[..., None] + [-1, 1] * u.deg).value,
                        # (d_roll[..., None] + [-2, 2] * u.deg).value,
                        (g_T[..., None] + [-5, 5] / u.mm).value,
                        # (d_inclination[..., None] + [-2, 2] * u.deg).value,
                        # (d_twist[..., None] + [-2, 2] * u.deg).value,
                        # (d_z[..., None] + [-10, 10] * u.mm).value,
                    ])
                    # simplex = 2 * np.random.random((7, 6)) - 1
                    # simplex = np.random.normal(0, 1, (9, 8))
                    # simplex /= np.sqrt(np.sum(np.square(simplex), ~0, keepdims=True))
                    # simplex *= 2
                    # simplex[..., :2] /= 40
                    # simplex[4] *= 3
                    # simplex[~0] *= 3

                    # simplex = np.array([
                    #     [-0.03, -0.03, -1, -2, -5, 2, 2, 10],
                    #     [0.03, -0.03, -1, -2, -5, -2, 2, 10],
                    #     [0.03, 0.03, -1, -2, -5, -2, -2, 10],
                    #     [0.03, 0.03, 1, -2, -5, -2, -2, -10],
                    #     [0.03, 0.03, 1, 2, -5, -2, -2, -10],
                    #     [-0.03, 0.03, 1, 2, 5, -2, -2, -10],
                    #     [-0.03, -0.03, 1, 2, 5, 2, -2, -10],
                    #     [-0.03, -0.03, -1, 2, 5, 2, 2, -10],
                    #     [-0.03, -0.03, -1, -2, 5, 2, 2, 10],
                    # ])
                    # simplex += x0
                    ref_image = None
                    images_masked = images
                else:
                    is_simple = True,
                    x0 = np.array([
                        g_twist.value,
                        g_inclination.value,
                        g_roll.value,
                        d_roll.value,
                        g_T.value,
                        d_inclination.value,
                        d_twist.value,
                        d_z.value,
                        # g_piston.value,
                        # v_0.value,
                        # v_x.value,
                        # v_y.value,
                        # stray_light.value,
                    ])
                    bounds = np.array([
                        (g_twist[..., None] + [-0.05, 0.05] * u.deg).value,
                        (g_inclination[..., None] + [-0.04, 0.04] * u.deg).value,
                        (g_roll[..., None] + [-1, 1] * u.deg).value,
                        (d_roll[..., None] + [-2, 2] * u.deg).value,
                        (g_T[..., None] + [-5, 5] / u.mm).value,
                        (d_inclination[..., None] + [-2, 2] * u.deg).value,
                        (d_twist[..., None] + [-2, 2] * u.deg).value,
                        (d_z[..., None] + [-10, 10] * u.mm).value,
                        # (g_piston[..., None] + [-1, 1] * u.mm).value,
                        # (v_0[..., None] + [-0.5, 0.5] * u.dimensionless_unscaled).value,
                        # (v_x[..., None] + [-0.001, 0.001] / u.arcsec).value,
                        # (v_y[..., None] + [-0.001, 0.001] / u.arcsec).value,
                        # ([0, 20] * u.adu).value,

                    ])
                    # simplex = 2 * np.random.random((7, 6)) - 1
                    # simplex /= 10
                    # simplex[..., :2] /= 10
                    # simplex += x0
                    rays = other.rays_output
                    distortion = rays.distortion(polynomial_degree=2)
                    wavelength = distortion.wavelength[..., ::2, 0, 0]
                    spatial_domain = oversize_ratio * u.Quantity([other.system.field_min, other.system.field_max])[..., :2]
                    pixel_domain = ([[0, 0], images.shape[~1:]] * u.pix)
                    w584 = distortion.wavelength[..., 1, 0, 0]
                    w584 = np.broadcast_to(w584[..., None], w584.shape + (2,), subok=True)
                    new_mask_584 = distortion.distort_cube(
                        cube=mask_584,
                        wavelength=w584,
                        spatial_domain_input=spatial_domain,
                        spatial_domain_output=pixel_domain,
                        spatial_samples_output=images.shape[~1:],
                    )
                    images_masked = images + new_mask_584
                    new_images = images_masked - other.stray_light[..., None, None, None]
                    new_images = distortion.distort_cube(
                        cube=new_images,
                        wavelength=wavelength,
                        spatial_domain_input=pixel_domain,
                        spatial_domain_output=spatial_domain,
                        spatial_samples_output=spatial_samples,
                        inverse=True,
                        fill_value=np.nan,
                    )
                    vignetting = rays.vignetting(polynomial_degree=1)
                    new_images = vignetting(
                        cube=new_images,
                        wavelength=wavelength,
                        spatial_domain=spatial_domain,
                        inverse=True,
                    )
                    new_images = optics.aberration.Vignetting.apply_model(
                        model=other.vignetting_correction,
                        cube=new_images,
                        wavelength=wavelength,
                        spatial_domain=spatial_domain,
                    )
                    # new_images = new_images / np.nanmean(new_images, axis=(~1, ~0))[..., None, None]
                    # new_images_lower = np.nanpercentile(new_images, 5, axis=(~1, ~0), keepdims=True)
                    # new_images_upper = np.nanpercentile(new_images, 95, axis=(~1, ~0), keepdims=True)
                    # new_images = (new_images - new_images_lower) / (new_images_upper - new_images_lower)
                    new_images[channel_index] = np.nan
                    ref_image = np.nanmean(new_images, axis=0)
                    # plt.figure()
                    # plt.imshow(ref_image[~0].value, vmin=0, vmax=200)
                    # ref_image = new_images[ref_index]
                    # ref_image = None
                    other.update()


                def cb(xk, convergence):
                    print(convergence)
                    other._rough_fit_func(xk, images, channel_index, ref_image, spatial_samples, oversize_ratio, is_simple, True)

                # result = scipy.optimize.shgo(
                # result = scipy.optimize.differential_evolution(
                result = scipy.optimize.minimize(
                # result = scipy.optimize.dual_annealing(
                    fun=other._rough_fit_func,
                    # func=other._rough_fit_func,
                    x0=x0,
                    bounds=bounds,
                    # method='L-BFGS-B',
                    options={
                        'disp': True,
                        # 'xtol': 1e-6,
                        'ftol': 1e-3,
                        # 'xatol': 1,
                        # 'fatol': 1e-3,
                        # 'initial_simplex': simplex,
                        # 'adaptive': True,
                    },
                    method='Powell',
                    # method='TNC',
                    # local_search_options={
                    #     'method': 'TNC',
                    #     'options': {
                    #         'gtol': 0.1,
                    #         'eps': 0.01,
                    #     }
                    #
                    # #     # 'adaptive': True,
                    # #     # 'gtol': 1e-3,
                    # #     'eps': 0.1,
                    # # #     # 'maxiter': 1,
                    # },
                    args=(
                        images_masked,
                        channel_index,
                        ref_image,
                        spatial_samples,
                        oversize_ratio,
                        is_simple,
                        plot_steps,
                        # aia_obs,
                    ),
                    # no_local_search=True,
                    # initial_temp=10,
                    # maxiter=15,
                    # visit=2,
                    # disp=True,
                    # polish=False,
                    # popsize=800,
                    # # strategy='currenttobest1exp',
                    # mutation=0.2,
                    # tol=0.001,
                    # workers=-1,
                    # callback=cb,
                )

                # print('DE Best result')
                # print(other._rough_fit_func(result.x, images.value, channel_index, spatial_samples))
                #
                if is_simple:
                    other = other._rough_fit_factory_simple(result.x, channel_index=channel_index)
                else:
                    other = other._rough_fit_factory(result.x, channel_index=channel_index)

        return other

    def _fit_factory(self, x: np.ndarray, ) -> 'Optics':
        other = self.copy()
        x = x.reshape((-1, 4))
        (
            d_roll,
            d_inclination,
            d_twist,
            # g_z,
            d_z,
            # g_T,
            # g_roll,
            # g_inclination,
            # d_r,
            # g_twist,
            # d_phi,
        ) = x
        # other.grating.roll = g_roll << u.deg
        # other.grating.inclination = g_inclination << u.deg
        # other.grating.twist = g_twist << u.deg
        # other.grating.piston = g_z << u.mm
        # other.grating.cylindrical_radius = x[16:20] << u.mm
        # other.grating.cylindrical_azimuth = x[20:24] << u.deg
        # other.detector.piston = x[24:28] << u.mm
        # other.detector.cylindrical_radius = d_r << u.mm
        # other.detector.cylindrical_azimuth = d_phi << u.deg
        other.detector.piston = d_z << u.mm
        other.detector.inclination = d_inclination << u.deg
        other.detector.roll = d_roll << u.deg
        other.detector.twist = d_twist << u.deg
        # other.grating.piston = x[28:32] * u.mm
        # other.wavelengths[~0] = x[32] * u.AA
        # other.grating.ruling_density = g_T / u.mm
        # other.grating.tangential_radius = x[32:36] << u.mm
        # other.grating.sagittal_radius = other.grating.tangential_radius
        # other.central_obscuration.position_error[vector.x] = x[0] << u.mm
        # other.central_obscuration.position_error[vector.y] = x[1] << u.mm
        # other.update()

        # dr = np.broadcast_to(self.detector.roll, (4,))
        # other.detector.roll[0] = dr[0]
        # other.detector.cylindrical_radius[ind] = self.detector.cylindrical_radius
        # other.detector.cylindrical_azimuth[ind] = self.detector.cylindrical_azimuth[ind]
        # other.detector.piston[ind] = self.detector.piston
        # other.detector.inclination[ind] = self.detector.inclination

        return other

    def _fit_func(
            self,
            x: np.ndarray,
            images: u.Quantity,
            spatial_samples: int = 100,
            plot_steps: bool = False,
            oversize_ratio: int = 1.1,
    ) -> float:

        other = self._fit_factory(x)

        images = images / np.median(images)

        new_images = other(
            data=images,
            wavelength=other.wavelength[::2],
            spatial_domain_input=[[0, 0], images.shape[~1:]] * u.pix,
            spatial_domain_output=oversize_ratio * u.Quantity([other.system.field_min, other.system.field_max]),
            spatial_samples_output=spatial_samples,
            inverse=True,
        )

        # distortion = other.rays_output.distortion(polynomial_degree=2)
        # wavelength = distortion.wavelength[:, ::2, 0, 0]
        # spatial_domain = oversize_ratio * u.Quantity([other.system.field_min, other.system.field_max])
        # pixel_domain = ([[0, 0], images.shape[~1:]] * u.pix)
        # new_images = distortion.distort_cube(
        #     cube=images,
        #     wavelength=wavelength,
        #     spatial_domain_input=pixel_domain,
        #     spatial_domain_output=spatial_domain,
        #     spatial_samples_output=spatial_samples,
        #     inverse=True,
        #     fill_value=np.nan,
        # )
        # vignetting = other.rays_output.vignetting(polynomial_degree=1)
        # new_images = vignetting(
        #     cube=new_images,
        #     wavelength=wavelength,
        #     spatial_domain=spatial_domain,
        #     inverse=True,
        # )
        new_images_nonan = np.nan_to_num(new_images)


        ish = 2 * (spatial_samples, )
        ix, iy = np.indices(ish)
        sx, sy = ish[vector.ix] // 2, ish[vector.iy] // 2
        ix, iy = ix - sx, iy - sy
        sx, sy = sx / oversize_ratio, sy / oversize_ratio
        sr = np.sqrt(sx * sx + sy * sy)
        ir = np.ones_like(ix, dtype=np.float)
        ir[iy > ix + sr] = 0
        ir[iy < ix - sr] = 0
        ir[iy > -ix + sr] = 0
        ir[iy < -ix - sr] = 0
        ir[ix > sx] = 0
        ir[ix < -sx] = 0
        ir[iy > sy] = 0
        ir[iy < -sy] = 0
        # ir = np.broadcast_to(ir, new_images.shape)
        # ir = ir * np.nanmedian(new_images, axis=(~1, ~0))[..., None, None]
        # ir = 2 - ir
        ir -= 1/2

        # new_images -= ir
        # new_images = np.abs(new_images)

        # new_images -= np.nanmedian(new_images, axis=(~1, ~0))[..., None, None]
        # ir -= ir.mean()

        # norm = -np.power(np.nanmean(new_images.prod((0, 1))), 1 / (new_images.shape[0] * new_images.shape[1]))
        # new_images = ir * new_images
        # norm = -np.nanmean(new_images)
        # new_images[ir == 0] = np.nan
        # new_images = new_images / np.nanmedian(new_images, axis=(~1, ~0))[..., None, None]
        # norm = -np.prod(new_images, axis=(0, 1)).mean()

        # base_norm = (new_images[::2] - new_images[1::2]).sum(0)
        # base_norm = (new_images - ir).sum(0)

        # norm = np.sqrt(np.nanmean(np.square(base_norm)))

        # plt.figure(figsize=(6, 6))
        # plt.imshow(base_norm[~0])
        # plt.show()

        # norm = np.sqrt(np.nanmean(np.square(new_images - ir)))
        # norm = np.sqrt(np.nanmean(np.square(new_images - np.roll(new_images, 1, axis=0))))
        # norm += np.sqrt(np.nanmean(np.square(new_images[0] - new_images[1])))
        # norm += np.sqrt(np.nanmean(np.square(new_images[2] - new_images[1])))
        # norm += np.sqrt(np.nanmean(np.square(new_images[3] - new_images[1])))
        # norm /= 4

        # n1 = np.roll(new_images, 1, axis=0)
        # n2 = np.roll(new_images, 2, axis=0)
        # n3 = np.roll(new_images, 3, axis=0)
        # norm = 3 * np.nanmean(np.square(new_images - ir))
        # norm += np.nanmean(np.square(n1 - new_images))
        # norm += np.nanmean(np.square(n2 - new_images))
        # norm += np.nanmean(np.square(n3 - new_images))



        ref_image = new_images_nonan[1]

        p11 = scipy.signal.correlate(new_images_nonan[0, 0], ref_image[0], mode='full') / ref_image.size
        p21 = scipy.signal.correlate(new_images_nonan[2, 0], ref_image[0], mode='full') / ref_image.size
        p31 = scipy.signal.correlate(new_images_nonan[3, 0], ref_image[0], mode='full') / ref_image.size
        p12 = scipy.signal.correlate(new_images_nonan[0, 1], ref_image[1], mode='full') / ref_image.size
        p22 = scipy.signal.correlate(new_images_nonan[2, 1], ref_image[1], mode='full') / ref_image.size
        p32 = scipy.signal.correlate(new_images_nonan[3, 1], ref_image[1], mode='full') / ref_image.size

        # p1 = p1.prod(0)
        # p2 = p2.prod(0)
        # p3 = p3.prod(0)

        # p3 = scipy.signal.correlate(p2, p1, mode='same')

        # pnorm = scipy.signal.correlate(1 + 0 * new_images[0], 1 + 0 * new_images[0], mode='same').mean(0) / new_images[0].size
        # p1 /= pnorm
        # p2 /= pnorm
        # p3 /= pnorm

        # p1 = scipy.signal.correlate(new_images, new_images, mode='full') / new_images.size
        # norm = p1.max((~1, ~0)).sum()

        p4 = p11 + p21 + p31 + p12 + p22 + p32
        # p2 = p1.max((0, 1))
        # p2 = scipy.signal.correlate(new_images, new_images, mode='same')
        # p3 = scipy.signal.correlate(new_images, new_images, mode='same')
        # p4 = p1 * p2 * p3
        # p5 = p4.prod(0)

        # prenorm = ir * (new_images[3] - new_images[2] + new_images[1] - new_images[0])
        norm = -(p11.max() + p21.max() + p31.max() + p12.max() + p22.max() + p32.max()) / 6

        lag11 = np.array(np.unravel_index(np.argmax(p11), p11.shape)) - np.array(p11.shape) // 2
        lag21 = np.array(np.unravel_index(np.argmax(p21), p21.shape)) - np.array(p21.shape) // 2
        lag31 = np.array(np.unravel_index(np.argmax(p31), p31.shape)) - np.array(p31.shape) // 2
        lag12 = np.array(np.unravel_index(np.argmax(p12), p12.shape)) - np.array(p12.shape) // 2
        lag22 = np.array(np.unravel_index(np.argmax(p22), p22.shape)) - np.array(p22.shape) // 2
        lag32 = np.array(np.unravel_index(np.argmax(p32), p32.shape)) - np.array(p32.shape) // 2

        new_images[0, 0] = np.roll(new_images[0, 0], -lag11, axis=(~1, ~0))
        new_images[2, 0] = np.roll(new_images[2, 0], -lag21, axis=(~1, ~0))
        new_images[3, 0] = np.roll(new_images[3, 0], -lag31, axis=(~1, ~0))
        new_images[0, 1] = np.roll(new_images[0, 1], -lag12, axis=(~1, ~0))
        new_images[2, 1] = np.roll(new_images[2, 1], -lag22, axis=(~1, ~0))
        new_images[3, 1] = np.roll(new_images[3, 1], -lag32, axis=(~1, ~0))

        d1 = np.nanmean(np.square(new_images[0] - new_images[1]))
        d2 = np.nanmean(np.square(new_images[2] - new_images[1]))
        d3 = np.nanmean(np.square(new_images[3] - new_images[1]))
        norm = np.sqrt((d1 + d2 + d3) / 3)

        if plot_steps:

            norm_base = new_images[3] - new_images[2] + new_images[1] - new_images[0]
            norm_base = norm_base.value
            _, axs = plt.subplots(ncols=2, figsize=(12, 6))
            axs[0].imshow(norm_base[0], vmin=np.nanpercentile(norm_base[0], 2), vmax=np.nanpercentile(norm_base[0], 98))
            axs[1].imshow(norm_base[1], vmin=np.nanpercentile(norm_base[1], 2), vmax=np.nanpercentile(norm_base[1], 98))
            plt.show()


            print('detector.roll', other.detector.roll)
            print('detector.inclination', other.detector.inclination)
            print('detector.twist', other.detector.twist)
            # print('grating.piston', other.grating.piston)
            print('detector.piston', other.detector.piston)
            # print('grating.ruling_density', other.grating.ruling_density)
            # print('grating.roll', other.grating.roll)
            # print('grating.inclination', other.grating.inclination)
            # print('detector.cylindrical_radius', other.detector.cylindrical_radius)
            # print('grating.twist', other.grating.twist)
            # print('detector.cylindrical_azimuth', other.detector.cylindrical_azimuth)
            print('lags1', lag11, lag21, lag31, )
            print('lags2', lag12, lag22, lag32, )
            print(norm)
            print()



        # nw = np.roll(new_images, 1, axis=1)
        # norm += np.nanmean(np.square(n1 - nw))
        # norm += np.nanmean(np.square(n2 - nw))
        # norm += np.nanmean(np.square(n3 - nw))
        # norm += np.sqrt(np.nanmean(np.square(new_images - ir)))
        # norm /= 6
        # norm = np.sqrt(norm)
        # norm /= 3



        # print('grating.piston', other.grating.piston)
        # print('grating.cylindrical_radius', other.grating.cylindrical_radius)
        # print('grating.cylindrical_azimuth', other.grating.cylindrical_azimuth)
        # print('detector.cylindrical_radius', other.detector.cylindrical_radius)
        # print('detector.cylindrical azimuth', other.detector.cylindrical_azimuth)
        # print('OV wavelength', other.wavelengths[~0])
        # print('grating.radius', other.grating.tangential_radius)
        # print('grating.ruling_density', other.grating.ruling_density)
        # print('vignetting.x', vig_model.coefficients[2].flatten())
        # print('vignetting.y', vig_model.coefficients[3].flatten())
        # print('central_obscuration.position_error', other.central_obscuration.position_error)


        # plt.show()

        return norm

    def fit_to_images(
            self,
            images: u.Quantity,
            global_search: bool = True,
            local_search: bool = True,
            global_samples: int = 128,
            local_samples: int = 256,
            plot_steps: bool = False,
            oversize_ratio: int = 1.1,
    ) -> 'Optics':

        images = np.broadcast_to(images[:, None], images.shape[:1] + (2,) + images.shape[1:], subok=True)

        sh = images.shape[:1]

        other = self.copy()

        g_roll = np.broadcast_to(other.grating.roll, sh, subok=True)
        g_inclination = np.broadcast_to(other.grating.inclination, sh, subok=True)
        g_twist = np.broadcast_to(other.grating.twist, sh, subok=True)
        # g_z = np.broadcast_to(other.grating.piston, sh, subok=True)
        # g_r = np.broadcast_to(other.grating.cylindrical_radius, sh, subok=True)
        # g_phi = np.broadcast_to(other.grating.cylindrical_azimuth, sh, subok=True)


        d_r = np.broadcast_to(other.detector.cylindrical_radius, sh, subok=True)
        d_phi = np.broadcast_to(other.detector.cylindrical_azimuth, sh, subok=True)
        d_z = np.broadcast_to(other.detector.piston, sh, subok=True)
        d_inclination = np.broadcast_to(other.detector.inclination, sh, subok=True)
        d_roll = np.broadcast_to(other.detector.roll, sh, subok=True)
        d_twist = np.broadcast_to(other.detector.twist, sh, subok=True)
        g_z = np.broadcast_to(other.grating.piston, sh, subok=True)
        g_T = np.broadcast_to(other.grating.ruling_density, sh, subok=True)
        # g_r = np.broadcast_to(other.grating.tangential_radius, sh, subok=True)
        # wavl = other.wavelengths[~0:]
        t_position = other.central_obscuration.position_error
        # vig_model = other.rays_output.vignetting(polynomial_degree=1).model(inverse=True)
        # v_x = vignetting.model(inverse=True)
        # v_x = np.broadcast_to(0 / u.percent / u.deg, sh, subok=True)
        # v_y = np.broadcast_to(0 / u.percent / u.deg, sh, subok=True)

        x0 = np.concatenate([

            d_roll.value,
            d_inclination.value,
            d_twist.value,
            # g_z.value,
            d_z.value,
            # g_T.value,
            # g_roll.value,
            # g_inclination.value,
            # d_r.value,
            # g_twist.value,
            # d_phi.value,









            # g_r.value,
            # wavl.value,
            # v_x.value,
            # v_y.value,
        ])
        bounds = np.concatenate([

            (d_roll[..., None] + [-1, 1] * u.deg).value,
            (d_inclination[..., None] + [-2, 2] * u.deg).value,
            (d_twist[..., None] + [-2, 2] * u.deg).value,
            # (g_z[..., None] + [-5, 5] * u.mm).value,
            (d_z[..., None] + [-10, 10] * u.mm).value,
            # (g_T[..., None] + [-5, 5] / u.mm).value,
            # (g_roll[..., None] + [-2, 2] * u.deg).value,
            # (g_inclination[..., None] + [-0.02, 0.02] * u.deg).value,
            # (d_r[..., None] + [-1, 1] * u.mm).value,
            # (g_twist[..., None] + [-0.02, 0.02] * u.deg).value,
            # (d_phi[..., None] + [-2, 2] * u.deg).value,




            # (g_r[..., None] + [-5, 5] * u.mm).value,
            # (g_phi[..., None] + [-2, 2] * u.deg).value,



            # (g_z[..., None] + [-0.1, 0.1] * u.mm).value,


            # (g_r[..., None] + [-5, 5] * u.mm).value,
            # (wavl[..., None] + [-1, 1] * u.AA).value,
            # (v_x[..., None] + [-1e-5, 1e-5] / u.percent / u.arcsec).to(0 / u.percent / u.deg).value,
            # (v_y[..., None] + [-1e-5, 1e-5] / u.percent / u.arcsec).to(0 / u.percent / u.deg).value,
        ])
        if global_search:
            def cb(xk, convergence):
                print(convergence)
                other._fit_func(xk, images, global_samples, True)
            result = scipy.optimize.differential_evolution(
            # result = scipy.optimize.shgo(
                func=other._fit_func,
                bounds=bounds,
                args=(images, global_samples, plot_steps, oversize_ratio),
                # options={
                #     'disp': True
                # }
                disp=True,
                # mutation=0.1,
                # polish=False,
                popsize=60,
                workers=-1,
                callback=cb,
                tol=1e-3,
            )
            x0 = result.x
        if local_search:
            # rel_step = 3e-2
            # step_size = rel_step * (bounds[..., 1] - bounds[..., 0])
            result = scipy.optimize.minimize(
                fun=other._fit_func,
                x0=x0,
                bounds=bounds,
                # method='Powell',
                method='L-BFGS-B',
                options={
                    'disp': True,
                    # 'xtol': 1e-4,
                    'ftol': 1e-4,
                    'maxcor': 1000,
                    # 'maxiter': 0,
                    # 'gtol': 1e-2,
                    'eps': 1e-2,
                    # 'finite-diff_rel_step': 0.1
                },
                callback=lambda xk: other._fit_func(xk, images, local_samples, True, oversize_ratio),
                args=(images, local_samples, plot_steps, oversize_ratio),
            )

        return other._fit_factory(result.x)

    def _fit_grating_roll_ruling_factory(self, x: np.ndarray, ) -> 'Optics':
        other = self.copy()
        x = x.reshape((-1, 4))
        (
            g_roll,
            g_T,
            # d_roll,
        ) = x
        other.grating.roll = g_roll << u.deg
        other.grating.ruling_density = g_T / u.mm
        # other.detector.roll = d_roll << u.deg
        return other

    def _fit_grating_roll_ruling_func(
            self,
            x: np.ndarray,
            images: u.Quantity,
            spatial_samples: int = 100,
            plot_steps: bool = False,
            oversize_ratio: int = 1.1,
    ) -> float:

        other = self._fit_grating_roll_ruling_factory(x)

        images = images / np.median(images)

        new_images = other(
            data=images,
            wavelength=other.wavelength[::2],
            spatial_domain_input=[[0, 0], images.shape[~1:]] * u.pix,
            spatial_domain_output=oversize_ratio * u.Quantity([other.system.field_min, other.system.field_max]),
            spatial_samples_output=spatial_samples,
            inverse=True,
        )

        # distortion = other.rays_output.distortion(polynomial_degree=2)
        # wavelength = distortion.wavelength[:, ::2, 0, 0]
        # spatial_domain = oversize_ratio * u.Quantity([other.system.field_min, other.system.field_max])
        # pixel_domain = ([[0, 0], images.shape[~1:]] * u.pix)
        # new_images = distortion.distort_cube(
        #     cube=images,
        #     wavelength=wavelength,
        #     spatial_domain_input=pixel_domain,
        #     spatial_domain_output=spatial_domain,
        #     spatial_samples_output=spatial_samples,
        #     inverse=True,
        #     fill_value=np.nan,
        # )
        # vignetting = other.rays_output.vignetting(polynomial_degree=1)
        # new_images = vignetting(
        #     cube=new_images,
        #     wavelength=wavelength,
        #     spatial_domain=spatial_domain,
        #     inverse=True,
        # )
        new_images_nonan = np.nan_to_num(new_images)
        ref_image = new_images_nonan[1]

        # p11 = scipy.signal.correlate(new_images[0, 0], ref_image[0], mode='full') / ref_image.size
        # p21 = scipy.signal.correlate(new_images[2, 0], ref_image[0], mode='full') / ref_image.size
        # p31 = scipy.signal.correlate(new_images[3, 0], ref_image[0], mode='full') / ref_image.size
        # p12 = scipy.signal.correlate(new_images[0, 1], ref_image[1], mode='full') / ref_image.size
        # p22 = scipy.signal.correlate(new_images[2, 1], ref_image[1], mode='full') / ref_image.size
        # p32 = scipy.signal.correlate(new_images[3, 1], ref_image[1], mode='full') / ref_image.size
        #
        # p1 = (p11 + p12) / 2
        # p2 = (p21 + p22) / 2
        # p3 = (p31 + p32) / 2

        p1 = scipy.signal.correlate(new_images_nonan[0], ref_image, mode='same') / ref_image.size
        p2 = scipy.signal.correlate(new_images_nonan[2], ref_image, mode='same') / ref_image.size
        p3 = scipy.signal.correlate(new_images_nonan[3], ref_image, mode='same') / ref_image.size

        p1 = p1.prod(0)
        p2 = p2.prod(0)
        p3 = p3.prod(0)

        lag1 = np.array(np.unravel_index(np.argmax(p1), p1.shape)) - np.array(p1.shape) // 2
        lag2 = np.array(np.unravel_index(np.argmax(p2), p2.shape)) - np.array(p2.shape) // 2
        lag3 = np.array(np.unravel_index(np.argmax(p3), p3.shape)) - np.array(p3.shape) // 2

        new_images[0] = np.roll(new_images[0], -lag1, axis=(~1, ~0))
        new_images[2] = np.roll(new_images[2], -lag2, axis=(~1, ~0))
        new_images[3] = np.roll(new_images[3], -lag3, axis=(~1, ~0))

        d1 = np.nanmean(np.square(new_images[0] - new_images[1]))
        d2 = np.nanmean(np.square(new_images[2] - new_images[1]))
        d3 = np.nanmean(np.square(new_images[3] - new_images[1]))
        norm = np.sqrt((d1 + d2 + d3) / 3)

        if plot_steps:

            norm_base = new_images[3] - new_images[2] + new_images[1] - new_images[0]
            norm_base = norm_base.value
            _, axs = plt.subplots(ncols=2, figsize=(12, 6))
            axs[0].imshow(norm_base[0], vmin=np.nanpercentile(norm_base[0], 2), vmax=np.nanpercentile(norm_base[0], 98))
            axs[1].imshow(norm_base[1], vmin=np.nanpercentile(norm_base[1], 2), vmax=np.nanpercentile(norm_base[1], 98))
            plt.show()

        # norm = -(p1.max() + p2.max() + p3.max())

            print('grating.roll', other.grating.roll)
            print('grating.ruling_density', other.grating.ruling_density)
            print('lags', lag1, lag2, lag3)
            print(norm)
            print()

        return norm

    def fit_grating_roll_ruling_to_images(
            self,
            images: u.Quantity,
            spatial_samples: int = 512,
            plot_steps: bool = False,
            oversize_ratio: int = 1.1,
    ) -> 'Optics':

        images = np.broadcast_to(images[:, None], images.shape[:1] + (2,) + images.shape[1:], subok=True)

        sh = images.shape[:1]

        other = self.copy()

        g_roll = np.broadcast_to(other.grating.roll, sh, subok=True)
        g_T = np.broadcast_to(other.grating.ruling_density, sh, subok=True)
        d_roll = np.broadcast_to(other.detector.roll, sh, subok=True)

        x0 = np.concatenate([
            g_roll.value,
            g_T.value,
            # d_roll.value,
        ])
        bounds = np.concatenate([
            (g_roll[..., None] + [-2, 2] * u.deg).value,
            (g_T[..., None] + [-10, 10] / u.mm).value,
            # (g_roll[..., None] + [-2, 2] * u.deg).value,
        ])

        result = scipy.optimize.minimize(
            fun=other._fit_grating_roll_ruling_func,
            x0=x0,
            bounds=bounds,
            method='L-BFGS-B',
            options={
                'disp': True,
                # 'xtol': 1e-4,
                'ftol': 1e-4,
                'eps': 1e-2,
            },
            callback=lambda xk: other._fit_grating_roll_ruling_func(xk, images, spatial_samples, True, oversize_ratio),
            args=(images, spatial_samples, plot_steps, oversize_ratio),
        )

        return other._fit_grating_roll_ruling_factory(result.x)

    def _fit_grating_inclination_twist_factory(self, x: np.ndarray, ) -> 'Optics':
        other = self.copy()
        x = x.reshape((-1, 4))
        (
            g_inclination,
            g_twist,

        ) = x
        other.grating.inclination = g_inclination << u.deg
        other.grating.twist = g_twist << u.deg
        return other

    def _fit_grating_inclination_twist_func(
            self,
            x: np.ndarray,
            images: u.Quantity,
            spatial_samples: int = 100,
            plot_steps: bool = False,
            oversize_ratio: int = 1.1,
    ) -> float:

        other = self._fit_grating_inclination_twist_factory(x)

        images = images / np.median(images)

        new_images = other(
            data=images,
            wavelength=other.wavelength[::2],
            spatial_domain_input=[[0, 0], images.shape[~1:]] * u.pix,
            spatial_domain_output=oversize_ratio * u.Quantity([other.system.field_min, other.system.field_max]),
            spatial_samples_output=spatial_samples,
            inverse=True,
        )

        # distortion = other.rays_output.distortion(polynomial_degree=2)
        # wavelength = distortion.wavelength[:, ::2, 0, 0]
        # spatial_domain = oversize_ratio * u.Quantity([other.system.field_min, other.system.field_max])
        # pixel_domain = ([[0, 0], images.shape[~1:]] * u.pix)
        # new_images = distortion.distort_cube(
        #     cube=images,
        #     wavelength=wavelength,
        #     spatial_domain_input=pixel_domain,
        #     spatial_domain_output=spatial_domain,
        #     spatial_samples_output=spatial_samples,
        #     inverse=True,
        #     fill_value=np.nan,
        # )
        # vignetting = other.rays_output.vignetting(polynomial_degree=1)
        # new_images = vignetting(
        #     cube=new_images,
        #     wavelength=wavelength,
        #     spatial_domain=spatial_domain,
        #     inverse=True,
        # )
        # new_images_nonan = np.nan_to_num(new_images)
        # ref_image = new_images_nonan[1]
        #
        # p1 = scipy.signal.correlate(new_images_nonan[0], ref_image, mode='same') / ref_image.size
        # p2 = scipy.signal.correlate(new_images_nonan[2], ref_image, mode='same') / ref_image.size
        # p3 = scipy.signal.correlate(new_images_nonan[3], ref_image, mode='same') / ref_image.size
        #
        # p1 = p1.prod(0)
        # p2 = p2.prod(0)
        # p3 = p3.prod(0)
        #
        # lag1 = np.array(np.unravel_index(np.argmax(p1), p1.shape)) - np.array(p1.shape) // 2
        # lag2 = np.array(np.unravel_index(np.argmax(p2), p2.shape)) - np.array(p2.shape) // 2
        # lag3 = np.array(np.unravel_index(np.argmax(p3), p3.shape)) - np.array(p3.shape) // 2

        # p4 = scipy.signal.correlate(new_images, new_images, mode='same') / new_images.size
        # p4 = p4.prod((0, 1))
        # norm = -(p4.max())

        d1 = np.nanmean(np.square(new_images[0] - new_images[1]))
        d2 = np.nanmean(np.square(new_images[2] - new_images[1]))
        d3 = np.nanmean(np.square(new_images[3] - new_images[1]))
        norm = np.sqrt((d1 + d2 + d3) / 3)

        # d1 = np.sum(np.square(lag1), axis=0)
        # d2 = np.sum(np.square(lag2), axis=0)
        # d3 = np.sum(np.square(lag3), axis=0)
        # norm = np.sqrt((d1 + d2 + d3) / 3)


        # lag1 = np.array(np.unravel_index(np.argmax(p1), p1.shape)) - np.array(p1.shape) // 2
        # lag2 = np.array(np.unravel_index(np.argmax(p2), p2.shape)) - np.array(p2.shape) // 2
        # lag3 = np.array(np.unravel_index(np.argmax(p3), p3.shape)) - np.array(p3.shape) // 2

        if plot_steps:
            norm_base = new_images[3] - new_images[2] + new_images[1] - new_images[0]
            norm_base = norm_base.value
            _, axs = plt.subplots(ncols=2, figsize=(12, 6))
            axs[0].imshow(norm_base[0], vmin=np.nanpercentile(norm_base[0], 2), vmax=np.nanpercentile(norm_base[0], 98))
            axs[1].imshow(norm_base[1], vmin=np.nanpercentile(norm_base[1], 2), vmax=np.nanpercentile(norm_base[1], 98))
            plt.show()


            print('grating.inclination', other.grating.inclination - self.grating.inclination)
            print('grating.twist', other.grating.twist - self.grating.twist)
            print(norm)
            print()

        return norm

    def fit_grating_inclination_twist_to_images(
            self,
            images: u.Quantity,
            spatial_samples: int = 512,
            plot_steps: bool = False,
            oversize_ratio: int = 1.1,
    ) -> 'Optics':

        images = np.broadcast_to(images[:, None], images.shape[:1] + (2,) + images.shape[1:], subok=True)

        sh = images.shape[:1]

        other = self.copy()

        g_inclination = np.broadcast_to(other.grating.inclination, sh, subok=True)
        g_twist = np.broadcast_to(other.grating.twist, sh, subok=True)

        x0 = np.concatenate([
            g_inclination.value,
            g_twist.value,
        ])
        bounds = np.concatenate([
            (g_inclination[..., None] + [-0.06, 0.06] * u.deg).value,
            (g_twist[..., None] + [-0.15, 0.15] * u.deg).value,
        ])

        result = scipy.optimize.minimize(
            fun=other._fit_grating_inclination_twist_func,
            x0=x0,
            bounds=bounds,
            method='L-BFGS-B',
            options={
                'disp': True,
                # 'xtol': 1e-4,
                'ftol': 1e-4,
                'eps': 1e-4,
            },
            callback=lambda xk: other._fit_grating_inclination_twist_func(
                xk, images, spatial_samples, True, oversize_ratio),
            args=(images, spatial_samples, plot_steps, oversize_ratio),
        )

        return other._fit_grating_inclination_twist_factory(result.x)

    def fit_to_images_final(
            self,
            images: u.Quantity,
            spatial_samples: int = 512,
            num_iterations: int = 2,
            plot_steps: bool = False,
    ) -> 'Optics':
        other = self.copy()

        oversize_ratio = 1.3

        for i in range(num_iterations):
            other = other.fit_to_images(
                images=images, global_search=False, local_samples=spatial_samples, plot_steps=plot_steps,
                oversize_ratio=oversize_ratio)
            other = other.fit_grating_roll_ruling_to_images(
                images=images, spatial_samples=spatial_samples, plot_steps=plot_steps, oversize_ratio=oversize_ratio)
            other = other.fit_grating_inclination_twist_to_images(
                images=images, spatial_samples=spatial_samples, plot_steps=plot_steps, oversize_ratio=1.4)

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

        other.wavelength = u.Quantity([wavelength_1, (wavelength_1 + wavelength_2) / 2, wavelength_2])
        # other.wavelengths = u.Quantity([wavelength_1, wavelength_2])

        other.update()

        return other

    def plot_distance_annotations_zx(
            self,
            ax: matplotlib.axes.Axes,
            transform_extra: typ.Optional[kgpy.transform.rigid.TransformList] = None,
    ):
        with astropy.visualization.quantity_support():
            if transform_extra is None:
                transform_extra = kgpy.transform.rigid.TransformList()
            # transform_base = transform_extra + self.transform

            for surf in self.system.surfaces_all.flat_global:
                if surf.name == self.central_obscuration.name:
                    position_obscuration = (transform_extra + surf.transform)(vector.Vector3D.spatial())
                elif surf.name == self.primary.name:
                    position_primary = (transform_extra + surf.transform)(vector.Vector3D.spatial())
                elif surf.name == self.field_stop.name:
                    position_fs = (transform_extra + surf.transform)(vector.Vector3D.spatial())
                elif surf.name == self.grating.name:
                    position_grating = (transform_extra + surf.transform)(vector.Vector3D.spatial())
                elif surf.name == self.filter.name:
                    position_filter = (transform_extra + surf.transform)(vector.Vector3D.spatial())
                elif surf.name == self.detector.name:
                    position_detector = (transform_extra + surf.transform)(vector.Vector3D.spatial())

            # position_obscuration = (transform_base + self.central_obscuration.transform)(vector.Vector3D.spatial())
            # position_primary = (transform_base + self.primary.transform)(vector.Vector3D.spatial())
            # position_fs = (transform_base + self.field_stop.transform)(vector.Vector3D.spatial())
            # position_grating = (transform_base + self.grating.transform)(vector.Vector3D.spatial())
            # position_filter = (transform_base + self.filter.transform)(vector.Vector3D.spatial())
            # position_detector = (transform_base + self.detector.transform)(vector.Vector3D.spatial())

            # line_symmetry = ax.axhline(y=0, linestyle='--', color='gray')
            # text_symmetry = ax.text(
            #     x=position_fs.z / 3,
            #     y=0 * u.mm,
            #     s='axis of symmetry',
            #     horizontalalignment='center',
            #     verticalalignment='center',
            #     bbox=dict(
            #         facecolor='white',
            #         edgecolor='None',
            #     )
            # )

            font_size = matplotlib.rcParams['font.size']

            blended_transform_x = matplotlib.transforms.blended_transform_factory(ax.transData, ax.figure.dpi_scale_trans)

            annotation_primary_to_fs_x = plot.annotate_component(
                ax=ax,
                point_1=position_primary.zx,
                point_2=position_fs.zx,
                component='x',
                position_orthogonal=-0.2,
                transform=ax.get_xaxis_transform(),
            )
            annotation_fs_to_grating_x = plot.annotate_component(
                ax=ax,
                point_1=position_fs.zx,
                point_2=position_grating.zx,
                component='x',
                position_orthogonal=-0.2,
                transform=ax.get_xaxis_transform(),
            )
            annotation_fs_to_obscuration_x = plot.annotate_component(
                ax=ax,
                point_1=position_fs.zx,
                point_2=position_obscuration.zx,
                component='x',
                position_orthogonal=-0.3,
                # position_parallel=1,
                # horizontal_alignment='right',
                transform=ax.get_xaxis_transform(),
            )
            annotation_primary_to_grating_y = plot.annotate_component(
                ax=ax,
                point_1=position_primary.zx,
                point_2=position_grating.zx,
                component='y',
                position_orthogonal=-1550,
                position_parallel=0,
                vertical_alignment='top',
            )
            annotation_primary_to_filter_x = plot.annotate_component(
                ax=ax,
                point_1=position_primary.zx,
                point_2=position_filter.zx,
                component='x',
                position_orthogonal=-0.3,
                position_parallel=1,
                horizontal_alignment='right',
                transform=ax.get_xaxis_transform(),
            )
            annotation_primary_to_filter_y = plot.annotate_component(
                ax=ax,
                point_1=position_primary.zx,
                point_2=position_filter.zx,
                component='y',
                position_orthogonal=-650,
                position_parallel=0.3,
            )

            annotation_primary_to_detector_x = plot.annotate_component(
                ax=ax,
                point_1=position_primary.zx,
                point_2=position_detector.zx,
                component='x',
                position_orthogonal=-0.2,
                position_parallel=1,
                horizontal_alignment='left',
                transform=ax.get_xaxis_transform(),
            )
            annotation_primary_to_detector_y = plot.annotate_component(
                ax=ax,
                point_1=position_primary.zx,
                point_2=position_detector.zx,
                component='y',
                position_orthogonal=-400,
                position_parallel=0.4,
            )

            annotation_grating_tip = plot.annotate_angle(
                ax=ax,
                point_center=position_grating.zx,
                radius=10 * self.grating.surface.aperture_mechanical.max.y,
                angle_1=90 * u.deg,
                angle_2=90 * u.deg + self.grating.inclination,
                angle_label=90 * u.deg + self.grating.inclination / 2,
            )

    def plot_field_stop_projections(
            self,
            ax: matplotlib.axes.Axes,
    ):

        subsystem = optics.System(
            object_surface=self.field_stop.surface,
            surfaces=optics.surface.SurfaceList([
                self.grating.surface,
                self.detector.surface,
            ]),
            wavelength=self.wavelength,
            field_samples=self.field_samples,
            field_margin=1 * u.nm,
            field_is_stratified_random=self.field_is_stratified_random,
            pupil_samples=self.pupil_samples,
            pupil_is_stratified_random=self.pupil_is_stratified_random,
            grid_velocity_los=self.grid_velocity_los,
            pointing=self.pointing,
            roll=self.roll,
        )

        transform_detectors = kgpy.transform.rigid.TransformList([
            kgpy.transform.rigid.TiltZ(self.detector.cylindrical_azimuth),
            kgpy.transform.rigid.Translate(x=2.5 * self.detector.clear_half_width),
        ])

        self.detector.surface.plot(
            ax=ax,
            transform_extra=transform_detectors,
            plot_annotations=False,
            # to_global=True,
        )

        with astropy.visualization.quantity_support():

            colormap = plt.cm.viridis
            colornorm = plt.Normalize(vmin=self.wavelength.min().value, vmax=self.wavelength.max().value)

            wire = self.field_stop.surface.aperture.wire[..., np.newaxis, np.newaxis]
            wire.z = self.wavelength
            for w in range(wire.shape[~0]):
                ax.plot(
                    -4 * wire.x_final[..., w],
                    4 * wire.y_final[..., w],
                    # color=colormap(colornorm(self.wavelength[..., w].value)),
                    color='black',
                    # label=self.wavelength[..., w],
                )

            subsystem_model = subsystem.rays_output.distortion(polynomial_degree=2).model()
            wire = subsystem_model(wire)
            wire = transform_detectors(wire.to_3d(), num_extra_dims=3)

            for i in range(wire.shape[0]):
                for w in range(wire.shape[~0]):
                    if i == 0:
                        label_kwarg = dict(label=self.wavelength[..., w])
                    else:
                        label_kwarg = dict()
                    ax.plot(
                        wire.x[i, ..., w],
                        wire.y[i, ..., w],
                        color=colormap(colornorm(self.wavelength[..., w].value)),
                        **label_kwarg,
                    )

            length = self.field_stop.clear_radius / 3
            fiducial = vector.Vector3D.from_cylindrical(
                radius=length,
                azimuth=np.linspace(0, 360, 4, endpoint=True) * u.deg + 30 * u.deg,
                z=0 * u.mm
            )

            fiducial.x = fiducial.x[..., np.newaxis, np.newaxis]
            fiducial.y = fiducial.y[..., np.newaxis, np.newaxis]
            fiducial.y = fiducial.y - length
            fiducial.z = self.wavelength

            for w in range(fiducial.shape[~0]):
                ax.fill(
                    -4 * fiducial.x_final[..., w],
                    -4 * fiducial.y_final[..., w],
                    # color=colormap(colornorm(self.wavelength[..., w].value)),
                    color='black'
                )

            fiducial = subsystem_model(fiducial)
            fiducial = transform_detectors(fiducial.to_3d(), num_extra_dims=3)

            for i in range(fiducial.shape[0]):
                for w in range(fiducial.shape[~0]):
                    ax.fill(
                        fiducial.x[i, ..., w],
                        fiducial.y[i, ..., w],
                        color=colormap(colornorm(self.wavelength[..., w].value)),
                    )

            line = vector.Vector3D.spatial()
            line.y = length * [-1, 1]

            line.x = line.x[..., np.newaxis, np.newaxis]
            line.y = line.y[..., np.newaxis, np.newaxis]
            line.z = self.wavelength

            for w in range(line.shape[~0]):
                ax.plot(
                    -4 * line.x_final[..., w],
                    -4 * line.y_final[..., w],
                    # color=colormap(colornorm(self.wavelength[..., w].value)),
                    color='black'
                )

            line = subsystem_model(line)
            line = transform_detectors(line.to_3d(), num_extra_dims=3)

            for i in range(fiducial.shape[0]):
                for w in range(fiducial.shape[~0]):
                    ax.plot(
                        line.x[i, ..., w],
                        line.y[i, ..., w],
                        color=colormap(colornorm(self.wavelength[..., w].value)),
                    )