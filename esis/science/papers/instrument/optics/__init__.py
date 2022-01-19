import typing as typ
import pathlib
import numpy as np
import astropy.units as u
import esis

__all__ = ['as_designed_single_channel']


def _cache(factory: typ.Callable[[], esis.optics.Optics]):
    def func():
        path = pathlib.Path(__file__).parent / f'{factory.__name__}.pickle'
        if path.exists():
            optics = esis.optics.Optics.from_pickle(path)
        else:
            optics = factory()
            rays = optics.rays_output
            optics._bunch = None
            optics.to_pickle(path)

        return optics

    return func


default_pupil_samples = 15
default_pupil_is_stratified_random = True
default_field_samples = 15
default_field_is_stratified_random = False

default_kwargs = dict(
    pupil_samples=default_pupil_samples,
    pupil_is_stratified_random=default_pupil_is_stratified_random,
    field_samples=default_field_samples,
    field_is_stratified_random=default_field_is_stratified_random,
    all_channels=False,
)

num_emission_lines_default = 3


@_cache
def as_designed_single_channel() -> esis.optics.Optics:
    return esis.optics.design.final(
        pupil_samples=default_pupil_samples,
        pupil_is_stratified_random=default_pupil_is_stratified_random,
        field_samples=default_field_samples,
        field_is_stratified_random=default_field_is_stratified_random,
        all_channels=False,
    )


@_cache
def as_designed_active_channels() -> esis.optics.Optics:
    return esis.optics.design.final_active(
        pupil_samples=default_pupil_samples,
        pupil_is_stratified_random=default_pupil_is_stratified_random,
        field_samples=default_field_samples,
        field_is_stratified_random=default_field_is_stratified_random,
    )


@_cache
def as_measured_single_channel() -> esis.optics.Optics:
    return esis.flight.optics.as_measured(
        pupil_samples=default_pupil_samples,
        pupil_is_stratified_random=default_pupil_is_stratified_random,
        field_samples=default_field_samples,
        field_is_stratified_random=default_field_is_stratified_random,
        all_channels=False,
    )

error_pupil_samples = 11
error_pupil_is_stratified_random = False
error_field_samples = 11
error_field_is_stratified_random = False

error_kwargs = dict(
    pupil_samples=error_pupil_samples,
    pupil_is_stratified_random=error_pupil_is_stratified_random,
    field_samples=error_field_samples,
    field_is_stratified_random=error_field_is_stratified_random,
    all_channels=False,
)


@_cache
def error_optimized() -> esis.optics.Optics:
    return esis.optics.design.final(**error_kwargs).focus_and_align()


@_cache
def error_primary_decenter_x_min() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    r = opt_uncertainty.primary.translation_error.x.min()
    a = -opt.roll
    opt.primary.translation_error.x = r * np.cos(a)
    opt.primary.translation_error.y = r * np.sin(a)

    field_decenter = (np.arctan2(r, opt.primary.focal_length)).to(u.arcmin)
    opt.source.decenter.x = -field_decenter

    return opt.focus_and_align()


@_cache
def error_primary_decenter_x_max() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    r = opt_uncertainty.primary.translation_error.x.max()
    a = -opt.roll
    opt.primary.translation_error.x = r * np.cos(a)
    opt.primary.translation_error.y = r * np.sin(a)

    field_decenter = (np.arctan2(r, opt.primary.focal_length)).to(u.arcmin)
    opt.source.decenter.x = -field_decenter

    return opt.focus_and_align()


@_cache
def error_primary_decenter_y_min() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    r = opt_uncertainty.primary.translation_error.y.min()
    a = -opt.roll + 90 * u.deg
    opt.primary.translation_error.x = r * np.cos(a)
    opt.primary.translation_error.y = r * np.sin(a)

    field_decenter = (np.arctan2(r, opt.primary.focal_length)).to(u.arcmin)
    opt.source.decenter.y = field_decenter

    return opt.focus_and_align()


@_cache
def error_primary_decenter_y_max() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    r = opt_uncertainty.primary.translation_error.y.max()
    a = -opt.roll + 90 * u.deg
    opt.primary.translation_error.x = r * np.cos(a)
    opt.primary.translation_error.y = r * np.sin(a)

    field_decenter = (np.arctan2(r, opt.primary.focal_length)).to(u.arcmin)
    opt.source.decenter.y = field_decenter

    return opt.focus_and_align()


@_cache
def error_grating_translation_x_min() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    r = opt_uncertainty.grating.translation_error.x.min()
    a = -opt.roll
    opt.grating.translation_error.x = r * np.cos(a)
    opt.grating.translation_error.y = r * np.sin(a)
    return opt.focus_and_align()


@_cache
def error_grating_translation_x_max() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    r = opt_uncertainty.grating.translation_error.x.max()
    a = -opt.roll
    opt.grating.translation_error.x = r * np.cos(a)
    opt.grating.translation_error.y = r * np.sin(a)
    return opt.focus_and_align()


@_cache
def error_grating_translation_y_min() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    r = opt_uncertainty.grating.translation_error.y.min()
    a = -opt.roll + 90 * u.deg
    opt.grating.translation_error.x = r * np.cos(a)
    opt.grating.translation_error.y = r * np.sin(a)
    return opt.focus_and_align()


@_cache
def error_grating_translation_y_max() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    r = opt_uncertainty.grating.translation_error.y.max()
    a = -opt.roll + 90 * u.deg
    opt.grating.translation_error.x = r * np.cos(a)
    opt.grating.translation_error.y = r * np.sin(a)
    return opt.focus_and_align()


@_cache
def error_grating_translation_z_min() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    opt.grating.translation_error.z = opt_uncertainty.grating.translation_error.z.min()
    return opt.focus_and_align(focus_grating=False, focus_detector=True)


@_cache
def error_grating_translation_z_max() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    opt.grating.translation_error.z = opt_uncertainty.grating.translation_error.z.max()
    return opt.focus_and_align(focus_grating=False, focus_detector=True)


@_cache
def error_grating_roll_min() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    opt.grating.roll_error = opt_uncertainty.grating.roll_error.min()
    return opt.focus_and_align()


@_cache
def error_grating_roll_max() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    opt.grating.roll_error = opt_uncertainty.grating.roll_error.max()
    return opt.focus_and_align()


@_cache
def error_grating_radius_min() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    opt.grating.tangential_radius_error = opt_uncertainty.grating.tangential_radius_error.min()
    opt.grating.sagittal_radius_error = opt_uncertainty.grating.sagittal_radius_error.min()
    return opt.focus_and_align()


@_cache
def error_grating_radius_max() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    opt.grating.tangential_radius_error = opt_uncertainty.grating.tangential_radius_error.max()
    opt.grating.sagittal_radius_error = opt_uncertainty.grating.sagittal_radius_error.max()
    return opt.focus_and_align()


@_cache
def error_grating_ruling_density_min() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    opt.grating.ruling_density_error = opt_uncertainty.grating.ruling_density_error.min()
    return opt.focus_and_align()


@_cache
def error_grating_ruling_density_max() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    opt.grating.ruling_density_error = opt_uncertainty.grating.ruling_density_error.max()
    return opt.focus_and_align()


@_cache
def error_grating_ruling_spacing_linear_min() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    opt.grating.ruling_spacing_coeff_linear_error = opt_uncertainty.grating.ruling_spacing_coeff_linear_error.min()
    return opt.focus_and_align()


@_cache
def error_grating_ruling_spacing_linear_max() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    opt.grating.ruling_spacing_coeff_linear_error = opt_uncertainty.grating.ruling_spacing_coeff_linear_error.max()
    return opt.focus_and_align()


@_cache
def error_grating_ruling_spacing_quadratic_min() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    opt.grating.ruling_spacing_coeff_quadratic_error = opt_uncertainty.grating.ruling_spacing_coeff_quadratic_error.min()
    return opt.focus_and_align()


@_cache
def error_grating_ruling_spacing_quadratic_max() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    opt.grating.ruling_spacing_coeff_quadratic_error = opt_uncertainty.grating.ruling_spacing_coeff_quadratic_error.max()
    return opt.focus_and_align()


@_cache
def error_detector_translation_x_min() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    r = opt_uncertainty.detector.translation_error.x.min()
    a = -opt.roll
    opt.detector.translation_error.x = r * np.cos(a)
    opt.detector.translation_error.y = r * np.sin(a)
    return opt.focus_and_align()


@_cache
def error_detector_translation_x_max() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    r = opt_uncertainty.detector.translation_error.x.max()
    a = -opt.roll
    opt.detector.translation_error.x = r * np.cos(a)
    opt.detector.translation_error.y = r * np.sin(a)
    return opt.focus_and_align()


@_cache
def error_detector_translation_y_min() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    r = opt_uncertainty.detector.translation_error.y.min()
    a = -opt.roll + 90 * u.deg
    opt.detector.translation_error.x = r * np.cos(a)
    opt.detector.translation_error.y = r * np.sin(a)
    return opt.focus_and_align()


@_cache
def error_detector_translation_y_max() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    r = opt_uncertainty.detector.translation_error.y.max()
    a = -opt.roll + 90 * u.deg
    opt.detector.translation_error.x = r * np.cos(a)
    opt.detector.translation_error.y = r * np.sin(a)
    return opt.focus_and_align()


@_cache
def error_detector_translation_z_min() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    opt.detector.translation_error.z = opt_uncertainty.detector.translation_error.z.min()
    return opt.focus_and_align(focus_grating=False, focus_detector=False)


@_cache
def error_detector_translation_z_max() -> esis.optics.Optics:
    opt = esis.optics.design.final(**error_kwargs)
    opt_uncertainty = esis.optics.design.final(**error_kwargs, use_uncertainty=True)
    opt.detector.translation_error.z = opt_uncertainty.detector.translation_error.z.max()
    return opt.focus_and_align(focus_grating=False, focus_detector=False)
