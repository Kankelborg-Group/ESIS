from esis.data import data
from kgpy.io import fits
from esis.data.level_0 import Level0


def test_from_path():
    path = data.raw_path
    l0 = Level0.from_path(path)

    frame_paths = data.find_frames(path, data.num_channels)
    hdu = fits.load_hdu(frame_paths)
    print(repr(hdu[5,0].header))

    assert l0.data.size > 0
