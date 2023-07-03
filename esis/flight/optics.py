import typing as typ
import pathlib
import numpy as np
import esis.optics
import astropy.units as u
import kgpy.labeled
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
    primary_serial, primary_transmissivity = esis.optics.primary.efficiency.witness.vs_wavelength_recoat_1()
    opt.primary.material = kgpy.optics.surfaces.materials.MeasuredMultilayerMirror(
        plot_kwargs=opt.primary.material.plot_kwargs,
        name=primary_serial,
        thickness=opt.primary.material.thickness,
        cap=opt.primary.material.cap,
        main=opt.primary.material.main,
        base=opt.primary.material.base,
        transmissivity_function=primary_transmissivity,
    )

    opt.grating.serial_number = kgpy.labeled.Array(np.array([
        '89025',
        '89024',
        '89026',
        '89027',
    ], dtype=object), axes=['channel'])
    opt.grating.manufacturing_number = kgpy.labeled.Array(np.array([
        'UBO-16-024',
        'UBO-16-017',
        'UBO-16-019',
        'UBO-16-014',
    ], dtype=object), axes=['channel'])
    radius_014 = kgpy.labeled.Array([597.170, 597.210, 597.195] * u.mm, axes=['measurement'])
    radius_017 = kgpy.labeled.Array([597.065, 597.045, 597.050] * u.mm, axes=['measurement'])
    radius_019 = kgpy.labeled.Array([597.055, 597.045, 597.030] * u.mm, axes=['measurement'])
    radius_024 = kgpy.labeled.Array([596.890, 596.870, 596.880] * u.mm, axes=['measurement'])

    opt.grating.tangential_radius = np.stack([
        radius_024.mean(),
        radius_017.mean(),
        radius_019.mean(),
        radius_014.mean(),
    ], axis='channel')
    opt.grating.sagittal_radius = opt.grating.tangential_radius
    opt.grating.slope_error.value = kgpy.labeled.Array([2.05, 2.0, 2.05, 2.0] * u.urad, axes=['channel'])
    opt.grating.ripple.value = kgpy.labeled.Array([1.75, 1.75, 1.75, 1.5] * u.nm, axes=['channel'])
    opt.grating.microroughness.value = kgpy.labeled.Array([0.5, 0.45, 0.6, 0.65] * u.nm, axes=['channel'])
    grating_measurement = esis.optics.grating.efficiency.vs_wavelength()
    opt.grating.material = kgpy.optics.surfaces.materials.MeasuredMultilayerMirror(
        plot_kwargs=opt.grating.material.plot_kwargs,
        name='UBO-16-017',
        thickness=opt.grating.material.thickness,
        cap=opt.grating.material.cap,
        main=opt.grating.material.main,
        base=opt.grating.material.base,
        transmissivity_function=grating_measurement,
    )

    grating_serial, grating_witness_measurement = esis.optics.grating.efficiency.witness.vs_wavelength_g17()
    opt.grating.witness = kgpy.optics.surfaces.materials.MeasuredMultilayerMirror(
        plot_kwargs=opt.grating.material.plot_kwargs,
        name=grating_serial,
        thickness=opt.grating.material.thickness,
        cap=opt.grating.material.cap,
        main=opt.grating.material.main,
        base=opt.grating.material.base,
        transmissivity_function=grating_witness_measurement,
    )

    # opt.grating.tangential_radius = (597.46 * u.mm + 597.08 * u.mm) / 2
    # opt.grating.sagittal_radius = opt.grating.tangential_radius
    opt.grating.ruling_density = 2585.5 / u.mm

    opt.detector.serial_number = kgpy.labeled.Array(np.array([
        'SN6',
        'SN7',
        'SN9',
        'SN10'
    ], dtype=object), axes=['channel'])

    # numbers sourced from ESIS instrument paper as of 09/10/20
    opt.detector.gain = kgpy.labeled.Array([
        [2.57, 2.50, 2.52, 2.53],
        [2.55, 2.58, 2.57, 2.63],
        [2.57, 2.53, 2.52, 2.59],
        [2.60, 2.60, 2.54, 2.58],
    ] * u.electron / u.adu, axes=['channel', 'quadrant'])

    opt.detector.readout_noise = kgpy.labeled.Array([
        [3.9, 4.0, 4.1, 3.7],
        [3.9, 4.0, 4.0, 4.0],
        [4.1, 4.1, 4.1, 4.3],
        [3.9, 3.9, 4.2, 4.1],
    ] * u.adu, axes=['channel', 'quadrant'])

    opt.detector.dark_current = kgpy.labeled.Array(([
        [1.37e-4, 9.66e-5, 6.85e-5, 9.80e-5],
        [6.77e-5, 5.89e-5, 8.98e-5, 1.01e-4],
        [3.14e-5, 2.68e-5, 3.18e-5, 3.72e-5],
        [6.39e-4, 5.07e-5, 6.63e-5, 8.24e-5],
    ] * u.electron / u.ms).to(u.electron / u.s), axes=['channel', 'quadrant'])

    if not all_channels:
        chan_index = dict(channel=esis.optics.design.default_channel_active)
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
