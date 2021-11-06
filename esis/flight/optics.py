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
    kwargs = dict(
        pupil_samples=pupil_samples,
        pupil_is_stratified_random=pupil_is_stratified_random,
        field_samples=field_samples,
        field_is_stratified_random=field_is_stratified_random,
    )
    if all_channels:
        opt = esis.optics.design.final_active(**kwargs)
    else:
        opt = esis.optics.design.final(**kwargs, all_channels=all_channels, )

    opt.primary.radius = 2000 * u.mm
    opt.primary.slope_error.value = 0.436 * u.urad
    opt.primary.ripple.value = 0.938 * u.nm
    opt.primary.microroughness.value = ([5.38, 5.68] * u.AA).mean().to(u.nm)
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

    opt.grating.serial_number = np.array([
        '89025',
        '89024',
        '89026',
        '89027',
    ])
    opt.grating.manufacturing_number = np.array([
        'UBO-16-024',
        'UBO-16-017',
        'UBO-16-019',
        'UBO-16-014',
    ])
    radius_014 = [597.170, 597.210, 597.195] * u.mm
    radius_017 = [597.065, 597.045, 597.050] * u.mm
    radius_019 = [597.055, 597.045, 597.030] * u.mm
    radius_024 = [596.890, 596.870, 596.880] * u.mm
    opt.grating.tangential_radius = u.Quantity([
        radius_024.mean(),
        radius_017.mean(),
        radius_019.mean(),
        radius_014.mean(),
    ])
    opt.grating.sagittal_radius = opt.grating.tangential_radius
    opt.grating.slope_error.value = [2.05, 2.0, 2.05, 2.0] * u.urad
    opt.grating.ripple.value = [1.75, 1.75, 1.75, 1.5] * u.nm
    opt.grating.microroughness.value = [0.5, 0.45, 0.6, 0.65] * u.nm
    grating_measurement = esis.optics.grating.efficiency.vs_wavelength()
    grating_angle_input, grating_wavelength, grating_efficiency = grating_measurement
    opt.grating.material = kgpy.optics.surface.material.MeasuredMultilayerMirror(
        plot_kwargs=opt.grating.material.plot_kwargs,
        name='UBO-16-017',
        thickness=opt.grating.material.thickness,
        cap=opt.grating.material.cap,
        main=opt.grating.material.main,
        base=opt.grating.material.base,
        efficiency_data=grating_efficiency,
        wavelength_data=grating_wavelength,
    )

    grating_witness_measurement = esis.optics.grating.efficiency.witness.vs_wavelength_g17()
    (
        grating_serial,
        grating_witness_angle_input,
        grating_witness_wavelength,
        grating_witness_efficiency,
    ) = grating_witness_measurement
    opt.grating.witness = kgpy.optics.surface.material.MeasuredMultilayerMirror(
        plot_kwargs=opt.grating.material.plot_kwargs,
        name=grating_serial,
        thickness=opt.grating.material.thickness,
        cap=opt.grating.material.cap,
        main=opt.grating.material.main,
        base=opt.grating.material.base,
        efficiency_data=grating_witness_efficiency,
        wavelength_data=grating_witness_wavelength,
    )

    # opt.grating.tangential_radius = (597.46 * u.mm + 597.08 * u.mm) / 2
    # opt.grating.sagittal_radius = opt.grating.tangential_radius
    opt.grating.ruling_density = 2585.5 / u.mm

    opt.detector.serial_number = np.array([
        'SN6',
        'SN7',
        'SN9',
        'SN10'
    ])

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

    opt.detector.dark_current = ([
        [1.37e-4, 9.66e-5, 6.85e-5, 9.80e-5],
        [6.77e-5, 5.89e-5, 8.98e-5, 1.01e-4],
        [3.14e-5, 2.68e-5, 3.18e-5, 3.72e-5],
        [6.39e-4, 5.07e-5, 6.63e-5, 8.24e-5],
    ] * u.electron / u.ms).to(u.electron / u.s)

    if not all_channels:
        chan_index = esis.optics.design.default_channel_active
        opt.grating.serial_number = opt.grating.serial_number[chan_index]
        opt.grating.manufacturing_number = opt.grating.manufacturing_number[chan_index]
        opt.grating.tangential_radius = opt.grating.tangential_radius[chan_index]
        opt.grating.sagittal_radius = opt.grating.sagittal_radius[chan_index]
        opt.detector.gain = opt.detector.gain[chan_index]
        opt.detector.readout_noise = opt.detector.readout_noise[chan_index]

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
