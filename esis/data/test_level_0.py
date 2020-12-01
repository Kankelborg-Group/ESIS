from .. import flight
from . import Level_0
import esis.optics


class TestLevel_0:

    def test_from_directory(self):
        level_0 = Level_0.from_directory(flight.raw_img_dir, detector=flight.optics.as_measured().detector)
        assert len(level_0.intensity.shape) == 4
        assert level_0.intensity.sum() > 0
        assert level_0.intensity.sum() > 0
