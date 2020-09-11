import pathlib
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import astropy.io.fits
import astropy.time
import skimage.transform

import kso.kso_iii.engineering.esis.optics.components.detectors.sensor

from esis.data import data, level_1


def test_find_frames():

    frame_list = data.find_frames(data.raw_path, 4)

    assert frame_list.shape[0] > 0
    assert frame_list.shape[-1] == 4


def test_load_hdu():

    frame_list = data.find_frames(data.raw_path, data.num_channels)

    hdu = data.load_hdu(frame_list)

    assert hdu.shape[0] == frame_list.shape[0]
    assert hdu.shape[1] == frame_list.shape[1]
    assert isinstance(hdu.flat[0], astropy.io.fits.PrimaryHDU)


def test_extract_frames():

    frame_list = data.find_frames(data.raw_path, data.num_channels)

    hdu = data.load_hdu(frame_list)

    frames = data.extract_data(hdu)

    assert frames.shape[0] == hdu.shape[0]
    assert frames.shape[1] == hdu.shape[1]
    assert frames.shape[2] == kso.kso_iii.engineering.esis.optics.components.detectors.sensor.npix_raw_y
    assert frames.shape[3] == kso.kso_iii.engineering.esis.optics.components.detectors.sensor.npix_raw_x


def test_load_frames():

    frame_list = data.find_frames(data.raw_path, data.num_channels)

    frames = data.load_frames(frame_list)

    assert frames.shape[0] > 0
    assert frames.shape[1] == data.num_channels
    assert frames.shape[2] == kso.kso_iii.engineering.esis.optics.components.detectors.sensor.npix_raw_y
    assert frames.shape[3] == kso.kso_iii.engineering.esis.optics.components.detectors.sensor.npix_raw_x


def test_extract_header_value():

    frame_list = data.find_frames(data.raw_path, data.num_channels)

    hdu = data.load_hdu(frame_list)

    key = 'IMG_TS'

    ts = data.extract_header_value(hdu, key)

    assert hdu.shape == ts.shape
    assert isinstance(ts.item(0), str)


def test_extract_times():

    frame_path = data.find_frames(data.raw_path, data.num_channels)

    hdu = data.load_hdu(frame_path)

    times = data.extract_times(hdu, 'IMG_TS')

    assert times.shape == hdu.shape
    assert isinstance(times, astropy.time.Time)


def test_calc_dark():

    frame_list = data.find_frames(data.raw_path, data.num_channels)
    frames = data.load_frames(frame_list)

    start_ind, end_ind = data.signal_indices(frames)
    frames, darks = data.organize_array(frames, start_ind, end_ind)

    dark = data.calc_dark(darks)

    assert dark.shape[0] == 1
    assert dark.shape[1:] == darks.shape[1:]


def test_remove_dark():

    frame_list = data.find_frames(data.raw_path, data.num_channels)
    frames = data.load_frames(frame_list)

    start_ind, end_ind = data.signal_indices(frames)
    frames, darks = data.organize_array(frames, start_ind, end_ind)

    new_frames = data.remove_dark(frames.copy(), darks)

    assert new_frames.shape == frames.shape
    assert new_frames.sum() < frames.sum()


def test_remove_overscan_pixels():

    frame_list = data.find_frames(data.raw_path, data.num_channels)
    frames = data.load_frames(frame_list)

    new_frames = data.remove_overscan_pixels(frames, n_overscan_pix=kso.kso_iii.engineering.esis.optics.components.detectors.sensor.npix_overscan)

    assert new_frames.shape[:~0] == frames.shape[:~0]
    assert new_frames.shape[~0] == frames.shape[~0] - 2 * kso.kso_iii.engineering.esis.optics.components.detectors.sensor.npix_overscan

    assert new_frames.sum() > 0


def test_remove_blank_pixels():

    frame_list = data.find_frames(data.raw_path, data.num_channels)
    frames = data.load_frames(frame_list)

    new_frames = data.remove_blank_pixels(frames, n_blank_pixels=kso.kso_iii.engineering.esis.optics.components.detectors.sensor.npix_blank)

    assert new_frames.shape[:~0] == frames.shape[:~0]
    assert new_frames.shape[~0] == frames.shape[~0] - 2 * kso.kso_iii.engineering.esis.optics.components.detectors.sensor.npix_blank

    assert new_frames.sum() > 0


def test_organize_frames():

    frame_list = data.find_frames(data.raw_path, 4)
    frames = data.load_frames(frame_list)

    start_ind, end_ind = data.signal_indices(frames)
    frames, darks = data.organize_array(frames, start_ind, end_ind)

    assert frames.shape[0] > 0
    assert darks.shape[0] > 0
    assert frames.shape[1:] == darks.shape[1:]


def test_prep_to_view():

    rotation_angles = 45 * np.arange(4) * u.deg

    # frame_list = data.find_frames(data.raw_path, data.num_channels)
    # frames = data.load_frames(frame_list)
    path = pathlib.Path(__file__).parent / 'esis_Level1.pickle'
    esis = level_1.Level_1.from_pickle(path)

    theta = 12.252

    img_transform = skimage.transform.AffineTransform(scale=(1 / np.cos(np.radians(theta)), 1), rotation=0)

    for i in range(esis.intensity.shape[0]):
        for j in range(esis.intensity.shape[1]):
            esis.intensity[i, j, ...] = skimage.transform.warp(esis.intensity[i, j, ...], img_transform)

    frames = esis.intensity
    frames = data.prep_to_view(frames, rotation_angles)

    ani = data.animate(frames)

    plt.rcParams['animation.embed_limit'] = 100.0

    with open('test.html', 'w') as f:
        f.write(ani.to_jshtml(fps=10))


def test_animate():

    pass

def test_remove_bias():
    n_blank_pixels = kso.kso_iii.engineering.esis.optics.components.detectors.sensor.npix_blank
    n_overscan_pix = kso.kso_iii.engineering.esis.optics.components.detectors.sensor.npix_overscan

    frame_list = data.find_frames(data.raw_path, data.num_channels)
    frames = data.load_frames(frame_list)
    data.remove_bias(frames, n_overscan_pix, n_blank_pixels)

    scale = np.percentile(frames,99, axis=(-2, -1))

    sequence = 18

    plt.figure()
    fig = plt.imshow(frames[sequence,0],vmax=scale[sequence,1])
    plt.show()


