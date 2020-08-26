import typing as typ
import dataclasses
import pathlib
import numpy as np
import pickle
from astropy.io import fits
import tarfile

from esis.data import data
from esis.data import level_0
from kgpy.img.masks import mask as img_mask
from kgpy.mixin import Pickleable


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
    def from_level_0(cls, obs: level_0.Level0, despike = False) -> 'Level1':

        start_ind, end_ind = data.signal_indices(obs.data)

        npix_overscan = 2
        npix_blank = 50
        lev1 = calc_level_1(obs.data,npix_overscan, npix_blank, start_ind, end_ind, despike = despike)
        frames, darks = lev1

        return cls(
            frames,
            darks,
            data.organize_array(obs.times, start_ind, end_ind)[0],
            data.organize_array(obs.requested_exposure_time, start_ind, end_ind)[0],
            data.organize_array(obs.cam_id, start_ind, end_ind)[0],
        )




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
        n_overscan_pix: int,
        n_blank_pix: int,
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
    :param n_overscan_pix: Number of overscan
    :param n_blank_pix:
    :param start_ind:
    :param end_ind:
    :return:
    """
    frames, darks = data.basic_prep(frames, n_overscan_pix, n_blank_pix, start_ind, end_ind)

    if despike == True:
        print('Despiking data, this will take a while ...')
        frames, mask, stats = data.despike(frames)

    frames = data.orient_to_observer(frames)

    return frames, darks


