import typing as typ
import dataclasses
import pathlib
import numpy as np
from astropy.io import fits
import tarfile

from esis.data import level_0
from kgpy.img.masks import mask as img_mask
from kgpy.mixin import Pickleable

import scipy.stats
import esis.optics
from kgpy.img import spikes


__all__ = ['Level1', 'calc_level_1']


@dataclasses.dataclass
class Level1(Pickleable):
    intensity: np.ndarray
    darks: np.ndarray
    start_time: np.ndarray
    exposure_length: np.ndarray
    cam_id: np.ndarray
    sequence_metadata: np.ndarray = None
    analog_metadata: np.ndarray = None

    @classmethod
    def from_level_0(cls, obs: level_0.Level0, detector: esis.optics.components.Detector, despike = False) -> 'Level1':

        start_ind, end_ind = Level1.signal_indices(obs.data)

        frames = Level1.remove_bias(obs.data, detector.npix_blank)
        frames = Level1.remove_inactive_pixels(frames, detector.npix_overscan, detector.npix_blank)
        frames, darks = Level1.organize_array(frames, start_ind, end_ind)
        frames = Level1.remove_dark(frames,darks)

        if despike == True:
            print('Despiking data, this will take a while ...')
            frames, mask, stats = Level1.despike(frames)

        #orient to observer
        frames = np.flip(frames, axis=-2)

        start_time = Level1.organize_array(obs.times, start_ind, end_ind)[0]
        exposure_length = Level1.organize_array(obs.requested_exposure_time, start_ind, end_ind)[0]
        cam_id = Level1.organize_array(obs.cam_id, start_ind, end_ind)[0]
        return cls(
            frames,
            darks,
            start_time,
            exposure_length,
            cam_id,
        )

    @staticmethod
    def signal_indices(frames: np.ndarray) -> typ.Tuple[int, int]:
        m = np.percentile(frames, 99, axis=(-2, -1))
        g = np.gradient(m, axis=0)

        num_border_frames = 4

        start_ind = np.argmax(g, axis=0) - num_border_frames
        end_ind = np.argmin(g, axis=0) + num_border_frames

        start_ind = scipy.stats.mode(start_ind)[0][0]
        end_ind = scipy.stats.mode(end_ind)[0][0]

        return start_ind, end_ind
    @staticmethod
    def remove_bias(frames: np.ndarray, n_blank_pix):
        s = [slice(None)] * frames.ndim
        s[-1] = slice(n_blank_pix, ~(n_blank_pix - 1))
        s = tuple(s)

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

    @staticmethod
    def remove_inactive_pixels(frames: np.ndarray, n_overscan_pix, n_blank_pix, axis: int = ~0):
        frames = Level1.remove_overscan_pixels(frames, n_overscan_pix, ccd_long_axis=axis)

        frames = Level1.remove_blank_pixels(frames, n_blank_pix, axis=axis)

        return frames

    @staticmethod
    def remove_blank_pixels(frames: np.ndarray, n_blank_pixels: int, axis: int = ~0):
        s = Level1.identify_blank_pixels(frames, n_blank_pixels, axis)

        return frames[s]

    @staticmethod
    def identify_blank_pixels(frames: np.ndarray, n_blank_pixels: int, axis: int = ~0):
        s = [slice(None)] * frames.ndim
        s[-1] = slice(n_blank_pixels, ~(n_blank_pixels - 1))
        s = tuple(s)

        return s

    @staticmethod
    def identify_overscan_pixels(
            frames: np.ndarray,
            n_overscan_pix: int,
            ccd_long_axis: int = ~0
    ) -> typ.Tuple[typ.Tuple[typ.Union[slice, int], ...], ...]:
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

    @staticmethod
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

        s0, s1 = Level1.identify_overscan_pixels(frames, n_overscan_pix, ccd_long_axis)

        return np.concatenate([frames[s0], frames[s1]], axis=ccd_long_axis)

    @staticmethod
    def organize_array(frames: np.ndarray, start_ind: int, end_ind: int) -> typ.Tuple[np.ndarray, np.ndarray]:
        dark1 = frames[:start_ind, ...]
        signal = frames[start_ind:end_ind, ...]
        dark2 = frames[end_ind:, ...]
        dark2 = dark2[:dark1.shape[0]]

        darks = np.concatenate([dark1, dark2], axis=0)

        return signal, darks


    @staticmethod
    def remove_dark(signal: np.ndarray, darks: np.ndarray,
                    axis: typ.Optional[typ.Union[int, typ.Tuple[int]]] = 0) -> np.ndarray:
        """
        Using a sequence of signal frames and a sequence of dark frames, update the signal frames with the average dark
        frame removed.
        :param signal: An array of signal frames
        :param darks: An array of dark frames
        :param axis: Axis index or indices to average over
        :return: The `signal` array with the average dark frame removed
        """

        dark = np.percentile(darks, 50, axis=axis, keepdims=True)

        signal = signal - dark

        return signal

    def despike(frames: np.ndarray) -> typ.Tuple[np.ndarray, np.ndarray, typ.List[spikes.Stats]]:

        return spikes.identify_and_fix(frames, axis=(0, 2, 3), percentile_threshold=(0, 99.9), poly_deg=1, )


    def create_mask(self, sequence):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        import matplotlib.colors as colors

        shp = self.intensity.shape
        poly_list = []
        for i in range(shp[1]):
            img = self.intensity[sequence,i]
            fig, ax = plt.subplots()
            ax.imshow(img, cmap='gray_r', norm=colors.SymLogNorm(1))

            poly = Polygon(img_mask.default_vertices(ax), animated=True, facecolor='none')
            ax.add_patch(poly)
            p = img_mask.PolygonInteractor(ax, poly)

            ax.set_title('Click and drag a vertex to move it. Press "i" and near line to insert. \n '
                         'Click and hold vertex then press "d" to delete. \n'
                         'Press "t" to hide vertices. Exit to move to next image.')
            plt.show()

            poly_list.append(poly)

        return poly_list

    @staticmethod
    def default_pickle_path() -> pathlib.Path:
        return pathlib.Path(__file__).parents[1] / 'flight/esis_Level1.pickle'

    @classmethod
    def from_pickle(cls, path: typ.Optional[pathlib.Path] = None) -> 'Level1':
        lev1 = super().from_pickle(path)
        return lev1

    def to_fits(self, path: pathlib.Path):

        path.mkdir(parents=True, exist_ok=True)

        for sequence in range(self.start_time.shape[0]):
            for camera in range(self.start_time.shape[1]):
                name = 'ESIS_Level1_' + str(self.start_time[sequence, camera]) + '_' + str(camera + 1) + '.fits'
                filename = path / name

                hdr = fits.Header()
                hdr['CAM_ID'] = self.cam_id[sequence, camera]
                hdr['DATE_OBS'] = str(self.start_time[sequence, camera])
                hdr['IMG_EXP'] = self.exposure_length[sequence, camera]

                hdul = fits.HDUList()

                hdul.append(fits.PrimaryHDU(np.array(self.intensity[sequence, camera, ...]), hdr))

                hdul.writeto(filename, overwrite=True)

        output_file = path.name + '.tar.gz'

        with tarfile.open(path.parent / output_file, "w:gz") as tar:
            tar.add(path, arcname=path.name)

        return







def calc_level_1(
        frames: np.ndarray,
        detector: esis.optics.components.Detector,
        start_ind: int,
        end_ind: int,
        despike = False,
) -> typ.Tuple[np.ndarray, np.ndarray]:
    """
    Apply the Level-1 correction to a sequence of ESIS frames.
    The sequence of frames is assumed to contain a set of dark frames, followed by a set of light frames, followed by
    another set of dark frames.
    The Level-1 correction consists of the following steps:
     - Basic preparation
      - Use some of the inactive pixels to subtract a bias value from all frames
      - Organize the frames into light and dark frames.
      - Subtract an average dark frame from all the light frames.
     - Despike the data to remove high-energy particle hits.
     - Reorient the frames so that they are displayed the same way an observer would see the sun.
    :param frames: The input ESIS frames.
    :param
    :param start_ind:
    :param end_ind:
    :return:
    """

    frames = Level1.remove_bias(frames, detector.npix_overscan, detector.npix_blank)
    frames = remove_inactive_pixels(frames, detector.npix_overscan, detector.npix_blank)
    frames, darks = organize_array(frames, start_ind, end_ind)
    frames = remove_dark(frames, darks)

    if despike == True:
        print('Despiking data, this will take a while ...')
        frames, mask, stats = data.despike(frames)

    frames = data.orient_to_observer(frames)

    return frames, darks


