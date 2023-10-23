import abc
import dataclasses
import astropy.units as u
import named_arrays as na
import optika
from . import mixins

__all__ = [
    "Detector",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractDetector(
    optika.mixins.Printable,
    optika.mixins.Rollable,
    optika.mixins.Yawable,
    optika.mixins.Pitchable,
    optika.mixins.Translatable,
    mixins.CylindricallyTransformable,
):
    @property
    @abc.abstractmethod
    def name_channel(self) -> str | na.AbstractScalar:
        """human-readable name of each channel of this detector array"""

    @property
    @abc.abstractmethod
    def manufacturer(self) -> str:
        """the name of the company that manufactured this device"""

    @property
    @abc.abstractmethod
    def serial_number(self) -> str:
        """the serial number of this device"""

    @property
    @abc.abstractmethod
    def width_pixels(self) -> u.Quantity | na.AbstractCartesian2dVectorArray:
        """the physical size of a pixel"""

    @property
    @abc.abstractmethod
    def shape_pixels(self) -> int | na.AbstractCartesian2dVectorArray:
        """number of pixels along each axis of the detector"""

    @property
    def width_clear(self) -> u.Quantity | na.Cartesian2dVectorArray:
        return (self.width_pixels * self.shape_pixels).to(u.mm)

    @property
    @abc.abstractmethod
    def num_columns_overscan(self) -> int:
        """number of overscan columns captured by this detector"""

    @property
    @abc.abstractmethod
    def num_columns_blank(self) -> int:
        """number of black columns captured by this detector"""

    @property
    def shape_pixels_all(self) -> na.Cartesian2dVectorArray:
        shape_pixels = na.asanyarray(
            a=self.shape_pixels,
            like=na.Cartesian2dVectorArray(),
        )
        overscan = 2 * self.num_columns_overscan
        blank = 2 * self.num_columns_blank
        return na.Cartesian2dVectorArray(
            x=shape_pixels.x + overscan + blank,
            y=shape_pixels.y,
        )

    @property
    @abc.abstractmethod
    def width_border(self) -> u.Quantity | na.AbstractScalar:
        """size of the border around the light-sensitive area"""

    @property
    def width_mechanical(self) -> u.Quantity | na.AbstractScalar:
        return self.width_clear + 2 * self.width_border

    @property
    @abc.abstractmethod
    def clearance(self) -> u.Quantity | na.AbstractScalar:
        """minimum distance between this component and other components"""

    @property
    def surface(self) -> optika.surfaces.Surface:
        return optika.surfaces.Surface(
            name="detector",
            aperture=optika.apertures.RectangularAperture(
                half_width=self.width_clear / 2,
                active=False,
            ),
            aperture_mechanical=optika.apertures.RectangularAperture(
                half_width=self.width_mechanical / 2
            ),
            transformation=self.transformation,
        )

    @property
    @abc.abstractmethod
    def position_image(self) -> u.Quantity | na.AbstractCartesian2dVectorArray:
        """nominal position of the center of the field of view"""

    @property
    @abc.abstractmethod
    def distance_focus(self) -> u.Quantity | na.AbstractScalar:
        """the distance the detector can be moved along the optic axis to focus it"""

    @property
    @abc.abstractmethod
    def temperature(self) -> u.Quantity | na.AbstractScalar:
        """temperature of this device"""

    @property
    @abc.abstractmethod
    def gain(self) -> u.Quantity | na.AbstractScalar:
        """gain value of the analog to digital converter"""

    @property
    @abc.abstractmethod
    def readout_noise(self) -> u.Quantity | na.AbstractScalar:
        """standard deviation of the readout noise"""

    @property
    @abc.abstractmethod
    def dark_current(self) -> u.Quantity | na.AbstractScalar:
        """dark current value for this detector"""

    @property
    @abc.abstractmethod
    def charge_diffusion(self) -> u.Quantity | na.AbstractScalar:
        """standard deviation of the charge diffusion kernel"""

    @property
    @abc.abstractmethod
    def time_transfer(self) -> u.Quantity | na.AbstractScalar:
        """time required to perform a frame transfer operation"""

    @property
    @abc.abstractmethod
    def time_readout(self) -> u.Quantity | na.AbstractScalar:
        """time required to perform a readout operation"""

    @property
    @abc.abstractmethod
    def time_exposure(self) -> u.Quantity | na.AbstractScalar:
        """the current exposure length"""

    @property
    @abc.abstractmethod
    def time_exposure_min(self) -> u.Quantity:
        """the minimum exposure length supported by this detector"""

    @property
    @abc.abstractmethod
    def time_exposure_max(self) -> u.Quantity:
        """the maximum exposure length supported by this detector"""

    @property
    @abc.abstractmethod
    def timedelta_exposure_min(self) -> u.Quantity:
        """the smalled amount the exposure length can be incremented by"""

    @property
    @abc.abstractmethod
    def timedelta_synchronization(self) -> u.Quantity:
        """the synchronization error between the different channels"""

    @property
    @abc.abstractmethod
    def bits_adc(self) -> int:
        """the number of bits supported by the analog-to-digital converter"""

    @property
    @abc.abstractmethod
    def channel_trigger(self) -> int:
        """the channel that strobes the sychronization trigger"""


@dataclasses.dataclass(eq=False, repr=False)
class Detector(
    AbstractDetector,
):
    name_channel: str | na.AbstractScalar = ""
    manufacturer: str = ""
    serial_number: str = ""
    width_pixels: u.Quantity | na.AbstractCartesian2dVectorArray = 0 * u.mm
    shape_pixels: int | na.AbstractCartesian2dVectorArray = 0
    num_columns_overscan: int = 0
    num_columns_blank: int = 0
    width_border: u.Quantity | na.ScalarArray = 0 * u.mm
    distance_radial: u.Quantity | na.AbstractScalar = 0 * u.mm
    azimuth: u.Quantity | na.AbstractScalar = 0 * u.deg
    translation: u.Quantity | na.AbstractCartesian3dVectorArray = 0 * u.mm
    pitch: u.Quantity | na.AbstractScalar = 0 * u.deg
    yaw: u.Quantity | na.AbstractScalar = 0 * u.deg
    roll: u.Quantity | na.AbstractScalar = 0 * u.deg
    clearance: u.Quantity | na.ScalarArray = 0 * u.mm
    position_image: u.Quantity | na.AbstractCartesian2dVectorArray = 0 * u.mm
    distance_focus: u.Quantity | na.AbstractScalar = 0 * u.mm
    temperature: u.Quantity | na.ScalarArray = 0 * u.K
    gain: u.Quantity | na.ScalarArray = 0 * u.electron / u.DN
    readout_noise: u.Quantity | na.ScalarArray = 0 * u.DN
    dark_current: u.Quantity | na.ScalarArray = 0 * u.electron / u.s
    charge_diffusion: u.Quantity | na.AbstractScalar = 0 * u.um
    time_transfer: u.Quantity | na.AbstractScalar = 0 * u.s
    time_readout: u.Quantity | na.AbstractScalar = 0 * u.s
    time_exposure: u.Quantity | na.AbstractScalar = 0 * u.s
    time_exposure_min: u.Quantity | na.AbstractScalar = 0 * u.s
    time_exposure_max: u.Quantity | na.AbstractScalar = 0 * u.s
    timedelta_exposure_min: u.Quantity | na.AbstractScalar = 0 * u.s
    timedelta_synchronization: u.Quantity | na.AbstractScalar = 0 * u.s
    bits_adc: int = 0
    channel_trigger: int = 0
