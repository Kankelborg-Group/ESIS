import typing as typ
import pathlib
import numpy as np
import esis.optics
import astropy.units as u
import esis.optics

__all__ = ['as_measured', 'as_flown']


def as_measured(
        pupil_samples: int = 10,
        field_samples: int = 10,
        all_channels: bool = True
) -> esis.optics.Optics:
    opt = esis.optics.design.final(pupil_samples, field_samples, all_channels)

    # opt.grating.tangential_radius = (597.46 * u.mm + 597.08 * u.mm) / 2
    # opt.grating.sagittal_radius = opt.grating.tangential_radius
    opt.grating.ruling_density = 2585.5 / u.mm

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
