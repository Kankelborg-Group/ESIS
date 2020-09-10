from esis.data import data
from kgpy.io import fits
from .. import flight
from . import Level_0


class TestLevel_0:

    def test_from_directory(self):

         l0 = Level_0.from_directory(flight.raw_img_dir)


def test_from_path():
    path = data.raw_path
    l0 = Level_0.from_path(path)

    frame_paths = data.find_frames(path, data.num_channels)
    hdu = fits.load_hdu(frame_paths)
    print(repr(hdu[5,0].header))

    assert l0.data.size > 0
