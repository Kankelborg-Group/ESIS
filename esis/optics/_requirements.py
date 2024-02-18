import dataclasses
import numpy as np
import astropy.units as u
import optika

__all__ = [
    "Requirements",
]


@dataclasses.dataclass
class Requirements(
    optika.mixins.Printable,
):
    """
    Dataclass for storing the requirements of the ESIS optical system.
    """

    resolution_spatial: u.Quantity
    """The required spatial resolution of the instrument."""

    resolution_spectral: u.Quantity
    """The required spectral resolution of the instrument."""

    fov: u.Quantity
    """The required field of view of the instrument."""

    snr: u.Quantity
    """The required signal-to-noise ratio of the instrument."""

    cadence: u.Quantity
    """The required cadence of the instrument."""

    length_observation: u.Quantity
    """The required amount of observing time."""

    @property
    def resolution_angular(self) -> u.Quantity:
        """
        The angular resolution of the instrument.
        """
        return np.arctan2(self.resolution_spatial, 1 * u.AU).to(u.arcsec)
