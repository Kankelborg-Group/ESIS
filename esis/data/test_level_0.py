from esis.data import data
from kgpy.io import fits
from .. import flight
from . import Level_0


class TestLevel_0:

    def test_from_directory(self):
        level_0 = Level_0.from_directory(flight.raw_img_dir)
        assert len(level_0.data.shape) == 4
        assert level_0.data.sum() > 0
        assert level_0.cam_id.sum() > 0
