import typing as tp
import pathlib
import numpy as np
import scipy.ndimage
import scipy.stats
import scipy.signal
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.animation
from kgpy.io import fits
import esis.optics

from kgpy.img import spikes

num_channels = 4

raw_path = pathlib.Path(__file__).parents[1] / 'flight/images/'


def find_frames(path: pathlib.Path, n_channels: int) -> np.ndarray:
    """
    For a given path, find all the fits files and arrange them into a 2D array with shape (num exposures, num channels)
    :param path: The path to search for fits files
    :param n_channels: The number of channels we expect for the data
    :return: A 2D array where each element is a pathlib object pointing to a fits file
    """

    fits_list = np.array(list(path.glob('*.fit*')))
    fits_list.sort()
    fits_list = fits_list.reshape((n_channels, -1))
    fits_list = fits_list.transpose()

    print('num frames', len(fits_list))

    return fits_list

def load_frames(frame_path: np.ndarray) -> np.ndarray:
    """
    Load data from an array of fits files.
    :param frame_path: A `pathlib.Path` array of fits files
    :return: All the data in the fits files concatenated together into a single array.
    """

    hdu = fits.load_hdu(frame_path)

    frames = fits.extract_data(hdu)

    return frames


def calc_dark(frames: np.ndarray, axis: tp.Optional[tp.Union[int, tp.Tuple[int]]] = 0) -> np.ndarray:
    """
    From a sequence of frames, calculate a dark frame along the specified axis
    :param frames: An array of real numbers representing the frames to be averaged over
    :param axis: An array of axis indices to operate on
    :return: An array of real numbers representing the dark frame
    """

    return np.percentile(frames, 50, axis=axis, keepdims=True)


def remove_dark(signal: np.ndarray, darks: np.ndarray,
                axis: tp.Optional[tp.Union[int, tp.Tuple[int]]] = 0) -> np.ndarray:
    """
    Using a sequence of signal frames and a sequence of dark frames, update the signal frames with the average dark
    frame removed.
    :param signal: An array of signal frames
    :param darks: An array of dark frames
    :param axis: Axis index or indices to average over
    :return: The `signal` array with the average dark frame removed
    """

    dark = calc_dark(darks, axis=axis)

    signal = signal - dark

    return signal


def remove_overscan_pixels(frames: np.ndarray, n_overscan_pix: int, ccd_long_axis: int = ~0):
    """
    Trim the overscan pixels from an array of ESIS images.
    The overscan pixels are in the center of the images, running perpendicular to the long axis of the CCD.
    They are the last pixels to be read out on each row of each quadrant.
    :param frames: A sequence of ESIS images
    :param n_overscan_pix: The number of overscan pixels to remove from each quadrant.
    :param ccd_long_axis: Axis index of the CCD's long axis.
    :return: A copy of the `frames` array with the overscan pixels removed.
    """

    s0, s1 = identify_overscan_pixels(frames, n_overscan_pix, ccd_long_axis)

    return np.concatenate([frames[s0], frames[s1]], axis=ccd_long_axis)


def identify_overscan_pixels(
        frames: np.ndarray,
        n_overscan_pix: int,
        ccd_long_axis: int = ~0
) -> tp.Tuple[tp.Tuple[tp.Union[slice, int], ...], ...]:
    """

    :param frames:
    :param n_overscan_pix:
    :param ccd_long_axis:
    :return:
    """
    s0 = [slice(None)] * frames.ndim
    s1 = [slice(None)] * frames.ndim

    half_len = frames.shape[ccd_long_axis] // 2
    new_half_len = half_len - n_overscan_pix

    s0[ccd_long_axis] = slice(None, new_half_len)
    s1[ccd_long_axis] = slice(~(new_half_len - 1), None)

    s0 = tuple(s0)
    s1 = tuple(s1)

    return s0, s1


def remove_blank_pixels(frames: np.ndarray, n_blank_pixels: int, axis: int = ~0):
    s = identify_blank_pixels(frames, n_blank_pixels, axis)

    return frames[s]


def identify_blank_pixels(frames: np.ndarray, n_blank_pixels: int, axis: int = ~0):
    s = [slice(None)] * frames.ndim
    s[axis] = slice(n_blank_pixels, ~(n_blank_pixels - 1))
    s = tuple(s)

    return s


def remove_inactive_pixels(frames: np.ndarray, n_overscan_pix, n_blank_pix, axis: int = ~0):
    frames = remove_overscan_pixels(frames, n_overscan_pix, ccd_long_axis=axis)

    frames = remove_blank_pixels(frames, n_blank_pix, axis=axis)

    return frames


def remove_bias(frames: np.ndarray, n_overscan_pix, n_blank_pix):
    s = identify_blank_pixels(frames, n_blank_pix)

    half_height = frames.shape[2] // 2
    half_width = frames.shape[3] // 2

    # make a slice for each ESIS quad
    quad_1 = tuple([slice(None), slice(None), slice(half_height), slice(half_width)])
    quad_2 = tuple([slice(None), slice(None), slice(half_height, None), slice(half_width)])
    quad_3 = tuple([slice(None), slice(None), slice(half_height), slice(half_width, None)])
    quad_4 = tuple([slice(None), slice(None), slice(half_height, None), slice(half_width, None)])

    # find a bias for each quadrant
    b_1 = np.median(frames[quad_1][:, :, :, 0:s[-1].start], axis=(-2, -1), keepdims=True)
    b_2 = np.median(frames[quad_2][:, :, :, 0:s[-1].start], axis=(-2, -1), keepdims=True)
    b_3 = np.median(frames[quad_3][:, :, :, s[-1].stop:], axis=(-2, -1), keepdims=True)
    b_4 = np.median(frames[quad_4][:, :, :, s[-1].stop:], axis=(-2, -1), keepdims=True)

    # subtract bias from each quadrant
    frames[quad_1] -= b_1
    frames[quad_2] -= b_2
    frames[quad_3] -= b_3
    frames[quad_4] -= b_4

    return frames


def despike(frames: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray, tp.List[spikes.Stats]]:

    return spikes.identify_and_fix(frames, axis=(0, 2, 3), percentile_threshold=(0, 99.9), poly_deg=1,)


def rotate(frames: np.ndarray, rotation_angles: u.Quantity) -> np.ndarray:
    f = frames

    f = np.swapaxes(f, 0, 1)

    for c in range(len(rotation_angles)):
        f[c] = scipy.ndimage.rotate(f[c], rotation_angles[c].value, axes=(-2, -1), reshape=False)

    f = np.swapaxes(f, 0, 1)

    return f


def split_horizontally(frames: np.ndarray) -> np.ndarray:
    sh = list(frames.shape)
    sh.insert(~0, 2)
    sh[~0] //= 2

    frames = np.reshape(frames, sh)

    frames = np.swapaxes(frames, ~2, ~1)

    return frames


def make_square(frames: np.ndarray) -> np.ndarray:
    max_img_sz = np.min(frames.shape[~1:])

    return frames[..., :max_img_sz, :max_img_sz]


def rearrange_and_flatten_to_3d(frames: np.ndarray) -> np.ndarray:
    n_columns = 2

    frames = np.swapaxes(frames, 1, 2)

    sh = list(frames.shape)
    sh[~2] //= n_columns
    sh[~1] *= n_columns
    frames = np.reshape(frames, sh)

    frames = np.swapaxes(frames, ~0, ~1)

    sh = list(frames.shape)
    sh[~2] //= n_columns
    sh[~1] *= n_columns
    frames = np.reshape(frames, sh)

    frames = np.swapaxes(frames, 1, 2)

    sh = list(frames.shape)
    sh[~2] //= n_columns
    sh[~1] *= n_columns
    frames = np.reshape(frames, sh)

    frames = np.swapaxes(frames, ~0, ~1)

    frames = np.squeeze(frames)

    return frames


def prep_to_view(frames: np.ndarray, rotation_angles: u.Quantity) -> np.ndarray:
    # start_ind, end_ind = signal_indices(frames)
    # frames, darks = organize_array(frames, start_ind, end_ind)
    #
    # frames = remove_dark(frames, darks)

    frames = split_horizontally(frames)

    frames = make_square(frames)
    frames = rotate(frames, rotation_angles)

    frames = rearrange_and_flatten_to_3d(frames)

    return frames


def organize_array(frames: np.ndarray, start_ind: int, end_ind: int) -> tp.Tuple[np.ndarray, np.ndarray]:
    dark1 = frames[:start_ind, ...]
    signal = frames[start_ind:end_ind, ...]
    dark2 = frames[end_ind:, ...]
    dark2 = dark2[:dark1.shape[0]]

    darks = np.concatenate([dark1, dark2], axis=0)

    return signal, darks


def signal_indices(frames: np.ndarray) -> tp.Tuple[int, int]:
    m = np.percentile(frames, 99, axis=(-2, -1))
    g = np.gradient(m, axis=0)

    num_border_frames = 4

    start_ind = np.argmax(g, axis=0) - num_border_frames
    end_ind = np.argmin(g, axis=0) + num_border_frames

    start_ind = scipy.stats.mode(start_ind)[0][0]
    end_ind = scipy.stats.mode(end_ind)[0][0]

    return start_ind, end_ind


def edge_filter(frames: np.ndarray, kernel_sz: int = 3):
    dx = np.gradient(frames, axis=~0)
    dy = np.gradient(frames, axis=~1)

    p = 1

    t_lower = 10
    t_upper = 100 - t_lower

    ax = dx > np.percentile(dx, t_upper, axis=~0, keepdims=True)
    ay = dy > np.percentile(dy, t_upper, axis=~1, keepdims=True)

    bx = dx < np.percentile(dx, t_lower, axis=~0, keepdims=True)
    by = dy < np.percentile(dy, t_lower, axis=~1, keepdims=True)

    u = 2
    ax[..., :, :u] = False
    ay[..., :u, :] = False

    bx[..., :, -u:] = False
    by[..., -u:, :] = False

    ix = np.argmax(ax, axis=~0)
    iy = np.argmax(ay, axis=~1)

    jx = np.argmax(np.flip(bx, axis=~0), axis=~0)
    jy = np.argmax(np.flip(by, axis=~1), axis=~1)

    jx = ~jx
    jy = ~jy

    res = np.zeros_like(frames)

    res[..., :, ix] += 1
    res[..., iy, :] += 1

    res[..., :, jx] += 1
    res[..., jy, :] += 1

    return res


def animate(frames: np.ndarray, figsize=(24, 12), subplot_kw = None, **kwargs) -> matplotlib.animation.FuncAnimation:
    fig, ax = plt.subplots(figsize=figsize, subplot_kw = subplot_kw)
    im = ax.imshow(frames[0, ], **kwargs)

    def anim_func(i: int):
        im.set_data(frames[i, ])
        return [im]

    return matplotlib.animation.FuncAnimation(fig, anim_func, frames=frames.shape[0], interval=200)


def basic_prep(frames: np.ndarray, detector: esis.optics.components.Detector, start_ind: int, end_ind: int):

    frames = remove_bias(frames, detector.npix_overscan, detector.npix_blank)
    frames = remove_inactive_pixels(frames, detector.npix_overscan, detector.npix_blank)
    frames, darks = organize_array(frames, start_ind, end_ind)
    frames = remove_dark(frames, darks)

    return frames, darks


def orient_to_observer(frames: np.ndarray):
    frames = np.flip(frames, axis=-2)
    return frames