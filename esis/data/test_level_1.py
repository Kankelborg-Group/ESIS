import numpy as np
import pathlib
import esis.optics.design
from esis.data import Level_1, Level_0


def test_to_from_pickle():
    """
    This tests from_level_0 which calls calc_level_1 and then creates and deletes a test.pickle
    """
    optics = esis.flight.optics.as_measured()
    detector = optics.detector
    lev0 = Level_0.from_directory(esis.flight.raw_img_dir, detector)
    lev1 = Level_1.from_level_0(lev0)

    path = pathlib.Path('test.pickle')
    lev1.to_pickle(path)
    assert path.exists()

    esis_fp = Level_1.from_pickle(path)
    assert esis_fp.intensity.sum() > 0

    path.unlink()



# These aren't tests but driver programs ... where to put these?

# def test_to_fits(capsys):
#     with capsys.disabled():
#     # lev0 = level_0.Level0.from_path(data.raw_path)
#     # esis = level_1.Level1.from_level_0(lev0)
#
#         esis_fp = level_1.Level1.from_pickle(level_1.default_pickle_path)
#         dir = pathlib.Path(__file__).parent / 'ESIS_Level1_Fits/'
#
#         esis_fp.to_fits(dir)
#
# def test_create_mgx_mask():
#     lev1 = level_1.Level1.from_pickle(level_1.default_pickle_path)
#     my_favorite_sequence = 25
#     mask_polygons = lev1.create_mgx_mask(my_favorite_sequence)
#     for i,poly in enumerate(mask_polygons):
#         file = pathlib.Path(__file__).parent / 'esis_cam{}_mgx_mask.csv'.format(i + 1)
#         np.savetxt(file, poly.get_xy(), delimiter=',')
#
# def test_create_hei_mask():
#     lev1 = level_1.Level1.from_pickle(level_1.default_pickle_path)
#     my_favorite_sequence = 25
#     mask_polygons = lev1.create_mgx_mask(my_favorite_sequence)
#     for i,poly in enumerate(mask_polygons):
#         file = pathlib.Path(__file__).parent / 'esis_cam{}_hei_mask.csv'.format(i + 1)
#         np.savetxt(file, poly.get_xy(), delimiter=',')
#
#

# These are broken because of an inability to install pythonocc-core on Linux (at least)

# from kso.kso_iii.engineering.esis.optics.components.detectors import detector
# import kso.kso_iii.engineering.esis.optics.components.detectors.sensor

# def test_from_level_0():
#     lev0 = level_0.Level0.from_path(data.raw_path)
#     esis = level_1.Level1.from_level_0(lev0)
#
#     assert esis.intensity.shape[0] == esis.start_time.shape[0]
#     assert esis.intensity.shape[0] == esis.exposure_length.shape[0]
#
#     assert esis.intensity.shape[~0] == kso.kso_iii.engineering.esis.optics.components.detectors.sensor.npix_x
#     assert esis.intensity.shape[~1] == kso.kso_iii.engineering.esis.optics.components.detectors.sensor.npix_y
#
#
# def test_calc_level_1():
#     frame_list = data.find_frames(data.raw_path, 4)
#     frames = data.load_frames(frame_list)
#
#     frames = frames[:, 0:1, ...]
#
#     start_ind, end_ind = data.signal_indices(frames)
#     new_frames, darks = level_1.calc_level_1(frames.copy(),
#                                              kso.kso_iii.engineering.esis.optics.components.detectors.sensor.npix_overscan,
#                                              kso.kso_iii.engineering.esis.optics.components.detectors.sensor.npix_blank,
#                                              start_ind, end_ind)
#
#     assert new_frames.shape[0] > 0
#     assert new_frames.shape[1] == frames.shape[1]
#     assert new_frames.shape[~1] == kso.kso_iii.engineering.esis.optics.components.detectors.sensor.npix_y
#     assert new_frames.shape[~0] == kso.kso_iii.engineering.esis.optics.components.detectors.sensor.npix_x
