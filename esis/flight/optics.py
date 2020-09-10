import esis.optics
import astropy.units as u
from esis.optics import Optics


def as_measured(pupil_samples: int = 10,
                field_samples: int = 10,
                all_channels: bool = True
                ) -> Optics:

    esis_as_measured = esis.optics.design.final(pupil_samples, field_samples, all_channels)

    # numbers sorced from ESIS instrument paper as of 09/10/20
    esis_as_measured.components.detector.gain_tap1 = [2.57, 2.55, 2.57, 2.60] * u.electron  # per DN
    esis_as_measured.components.detector.gain_tap2 = [2.50, 2.58, 2.53, 2.60] * u.electron
    esis_as_measured.components.detector.gain_tap3 = [2.52, 2.57, 2.52, 2.54] * u.electron
    esis_as_measured.components.detector.gain_tap4 = [2.53, 2.63, 2.59, 2.58] * u.electron

    esis_as_measured.components.detector.readout_noise_tap1 = [3.9, 3.9, 4.1, 3.9]
    esis_as_measured.components.detector.readout_noise_tap2 = [4.0, 4.0, 4.1, 3.9]
    esis_as_measured.components.detector.readout_noise_tap3 = [4.1, 4.0, 4.1, 4.2]
    esis_as_measured.components.detector.readout_noise_tap4 = [3.7, 4.0, 4.3, 4.1]

    return esis_as_measured


def as_flown(pupil_samples: int = 10,
             field_samples: int = 10,
             all_channels: bool = True) -> Optics:
    esis_as_flown = esis.optics.design.final(pupil_samples, field_samples, all_channels)

    return esis_as_flown
