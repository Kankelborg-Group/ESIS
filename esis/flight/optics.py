import typing as typ
import pathlib
import numpy as np
import esis.optics
import astropy.units as u
import kgpy.optics
import esis.optics

__all__ = ['as_measured', 'as_flown']


def as_measured(
        pupil_samples: int = 10,
        pupil_is_stratified_random: bool = False,
        field_samples: int = 10,
        field_is_stratified_random: bool = False,
        all_channels: bool = True,
) -> esis.optics.Optics:
    opt = esis.optics.design.final(
        pupil_samples=pupil_samples,
        pupil_is_stratified_random=pupil_is_stratified_random,
        field_samples=field_samples,
        field_is_stratified_random=field_is_stratified_random,
        all_channels=all_channels,
    )

    primary_witness = esis.optics.primary.efficiency.witness.vs_wavelength_recoat_1()
    primary_serial, primary_angle_input, primary_wavelength, primary_efficiency = primary_witness
    opt.primary.material = kgpy.optics.surface.material.MeasuredMultilayerMirror(
        plot_kwargs=opt.primary.material.plot_kwargs,
        name=primary_serial,
        thickness=opt.primary.material.thickness,
        cap=opt.primary.material.cap,
        main=opt.primary.material.main,
        base=opt.primary.material.base,
        efficiency_data=primary_efficiency,
        wavelength_data=primary_wavelength,
    )

    grating_measurement = esis.optics.grating.efficiency.vs_wavelength()
    grating_angle_input, grating_wavelength, grating_efficiency = grating_measurement
    opt.grating.material = kgpy.optics.surface.material.MeasuredMultilayerMirror(
        plot_kwargs=opt.grating.material.plot_kwargs,
        name='grating 017',
        thickness=opt.grating.material.thickness,
        cap=opt.grating.material.cap,
        main=opt.grating.material.main,
        base=opt.grating.material.base,
        efficiency_data=grating_efficiency,
        wavelength_data=grating_wavelength,
    )

    # opt.grating.tangential_radius = (597.46 * u.mm + 597.08 * u.mm) / 2
    # opt.grating.sagittal_radius = opt.grating.tangential_radius
    opt.grating.ruling_density = 2585.5 / u.mm

    if all_channels:
        i1 = 1
        i4 = 4 + 1

        opt.grating.cylindrical_azimuth = opt.grating.cylindrical_azimuth[i1:i4]
        opt.grating.plot_kwargs['linestyle'] = opt.grating.plot_kwargs['linestyle'][i1:i4]

        opt.filter.cylindrical_azimuth = opt.filter.cylindrical_azimuth[i1:i4]
        opt.filter.plot_kwargs['linestyle'] = opt.filter.plot_kwargs['linestyle'][i1:i4]

        opt.detector.cylindrical_azimuth = opt.detector.cylindrical_azimuth[i1:i4]
        opt.detector.plot_kwargs['linestyle'] = opt.detector.plot_kwargs['linestyle'][i1:i4]

    # numbers sourced from ESIS instrument paper as of 09/10/20
    opt.detector.gain = [
        [2.57, 2.50, 2.52, 2.53],
        [2.55, 2.58, 2.57, 2.63],
        [2.57, 2.53, 2.52, 2.59],
        [2.60, 2.60, 2.54, 2.58],
    ] * u.electron / u.adu

    opt.detector.readout_noise = [
        [3.9, 4.0, 4.1, 3.7],
        [3.9, 4.0, 4.0, 4.0],
        [4.1, 4.1, 4.1, 4.3],
        [3.9, 3.9, 4.2, 4.1],
    ] * u.adu

    if not all_channels:
        opt.detector.gain = opt.detector.gain[esis.optics.design.default_channel]
        opt.detector.readout_noise = opt.detector.readout_noise[esis.optics.design.default_channel]

    return opt


as_flown_cache = pathlib.Path(__file__).parent / 'optics_as_flown.pickle'


def as_flown(
        pupil_samples: int = 10,
        field_samples: int = 10,
        disk_cache: pathlib.Path = as_flown_cache
) -> esis.optics.Optics:
    from . import level_1

    if disk_cache.exists():
        return esis.optics.Optics.from_pickle(disk_cache)

    else:
        esis_measured = as_measured(pupil_samples, field_samples)

        test_seq_index = 17
        cube = level_1().intensity[test_seq_index]
        cube = np.swapaxes(cube, ~1, ~0)
        stray_bg = np.percentile(cube.mean(~1), 1, axis=~0)
        cube -= stray_bg[..., None, None]
        thresh = np.percentile(cube, 99.9)
        cube[cube > thresh] = thresh
        cube_mean = np.median(cube, axis=(~1, ~0))[..., None, None]
        cube = cube * np.median(cube) / cube_mean
        cube = cube[..., 32:, :]

        esis_fit = esis_measured.fit_to_images_final(cube, spatial_samples=1024, num_iterations=3)
        esis_fit.to_pickle(disk_cache)

        return esis_fit
