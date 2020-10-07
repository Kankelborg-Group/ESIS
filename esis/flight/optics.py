import esis.optics
import astropy.units as u
from esis.optics import Optics

__all__ = ['as_measured', 'as_flown']


def as_measured(
        pupil_samples: int = 10,
        field_samples: int = 10,
        all_channels: bool = True
) -> Optics:
    opt = esis.optics.design.final(pupil_samples, field_samples, all_channels)

    # opt.grating.tangential_radius = (597.46 * u.mm + 597.08 * u.mm) / 2
    # opt.grating.sagittal_radius = opt.grating.tangential_radius
    opt.grating.ruling_density = 2585.5 / u.mm

    # numbers sourced from ESIS instrument paper as of 09/10/20
    detector = opt.detector
    detector.gain_tap1 = [2.57, 2.55, 2.57, 2.60] * u.electron / u.ct
    detector.gain_tap2 = [2.50, 2.58, 2.53, 2.60] * u.electron / u.ct
    detector.gain_tap3 = [2.52, 2.57, 2.52, 2.54] * u.electron / u.ct
    detector.gain_tap4 = [2.53, 2.63, 2.59, 2.58] * u.electron / u.ct

    detector.readout_noise_tap1 = [3.9, 3.9, 4.1, 3.9] * u.ct
    detector.readout_noise_tap2 = [4.0, 4.0, 4.1, 3.9] * u.ct
    detector.readout_noise_tap3 = [4.1, 4.0, 4.1, 4.2] * u.ct
    detector.readout_noise_tap4 = [3.7, 4.0, 4.3, 4.1] * u.ct

    # opt.update()

    return opt


def as_flown(
        pupil_samples: int = 10,
        field_samples: int = 10,
        all_channels: bool = True
) -> Optics:
    esis_as_flown = esis.optics.design.final(pupil_samples, field_samples, all_channels)

    return esis_as_flown
