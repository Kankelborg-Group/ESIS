import pytest
import msfc_ccd
from msfc_ccd._images._tests.test_sensor_images import AbstractTestAbstractSensorData
import esis


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.flights.f1.data.level_0(),
    ],
)
class TestLevel_0(
    AbstractTestAbstractSensorData,
):
    def test_timeline(self, a: esis.data.Level_0):
        result = a.timeline
        if result is not None:
            assert isinstance(result, esis.nsroc.Timeline)
