import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import optika._tests.test_mixins
import esis
from . import test_mixins


class AbstractTestAbstractDetector(
    optika._tests.test_mixins.AbstractTestPrintable,
    optika._tests.test_mixins.AbstractTestRollable,
    optika._tests.test_mixins.AbstractTestYawable,
    optika._tests.test_mixins.AbstractTestPitchable,
    optika._tests.test_mixins.AbstractTestTranslatable,
    test_mixins.AbstractTestCylindricallyTransformable,
):
    def test_manufacturer(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        assert isinstance(a.manufacturer, str)

    def test_serial_number(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        assert isinstance(a.serial_number, str)

    def test_width_pixels(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.width_pixels
        assert na.unit_normalized(result).is_equivalent(u.mm)
        assert np.all(result >= 0)

    def test_shape_pixels(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.shape_pixels
        assert na.unit(result) is None
        assert result >= 0

    def test_width_clear(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.width_clear
        assert na.unit_normalized(result).is_equivalent(u.mm)
        assert np.all(result >= 0)

    def test_num_columns_overscan(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.num_columns_overscan
        assert isinstance(result, int)
        assert result >= 0

    def test_num_columns_overscan(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.num_columns_blank
        assert isinstance(result, int)
        assert result >= 0

    def test_shape_pixels_all(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.shape_pixels_all
        assert na.unit(result) is None
        assert result >= a.shape_pixels

    def test_width_border(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.width_border
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.mm)
        assert np.all(result >= 0)

    def test_clearance(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.clearance
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.mm)
        assert np.all(result >= 0)

    def test_position_image(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.position_image
        assert na.unit_normalized(result).is_equivalent(u.mm)

    def test_distance_focus(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.distance_focus
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.mm)
        assert np.all(result >= 0)

    def test_temperature(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.temperature
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.K)
        assert np.all(result >= 0 * u.K)

    def test_gain(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.gain
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.electron / u.DN)
        assert np.all(result >= 0)

    def test_readout_noise(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.readout_noise
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.DN)
        assert np.all(result >= 0)

    def test_dark_current(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.dark_current
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.electron / u.s)
        assert np.all(result >= 0)

    def test_charge_diffusion(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.charge_diffusion
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.mm)
        assert np.all(result >= 0)

    def test_time_transfer(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.time_transfer
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.s)
        assert np.all(result >= 0)

    def test_time_readout(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.time_readout
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.s)
        assert np.all(result >= 0)

    def test_time_exposure(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.time_exposure
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.s)
        assert np.all(result >= 0)

    def test_time_exposure_min(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.time_exposure_min
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.s)
        assert np.all(result >= 0)

    def test_time_exposure_max(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.time_exposure_max
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.s)
        assert np.all(result >= 0)

    def test_timedelta_exposure_min(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.timedelta_exposure_min
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.s)
        assert np.all(result >= 0)

    def test_timedelta_synchronization(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.timedelta_synchronization
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.s)
        assert np.all(result >= 0)

    def bits_adc(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.bits_adc
        assert isinstance(result, int)
        assert result >= 0

    def channel_trigger(
        self,
        a: esis.optics.abc.AbstractDetector,
    ):
        result = a.channel_trigger
        assert isinstance(result, int)
        assert result >= 0


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.optics.Detector(),
        esis.optics.Detector(
            manufacturer="acme",
            serial_number="123abc",
            width_pixels=15 * u.um,
            shape_pixels=na.Cartesian2dVectorArray(2048, 1024),
            num_columns_overscan=5,
            num_columns_blank=50,
            distance_radial=150 * u.mm,
            azimuth=45 * u.deg,
            yaw=10 * u.deg,
            position_image=na.Cartesian2dVectorArray(10, 0) * u.mm,
            time_exposure=10 * u.s,
        ),
    ],
)
class TestDetector(AbstractTestAbstractDetector):
    pass
