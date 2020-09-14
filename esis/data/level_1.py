import typing as typ
import dataclasses
import warnings
import pathlib
import numpy as np
from astropy.io import fits
import tarfile
import astropy.units as u
import astropy.time
from kgpy import mixin, img
from kgpy.img.masks import mask as img_mask
import esis.optics
from . import Level_0


__all__ = ['Level_1']


@dataclasses.dataclass
class Level_1(mixin.Pickleable):
    intensity: u.Quantity
    start_time: astropy.time.Time
    exposure_length: u.Quantity
    cam_id: u.Quantity
    detector: esis.optics.components.Detector
    sequence_metadata: np.ndarray = None
    analog_metadata: np.ndarray = None

    @classmethod
    def from_level_0(cls, lev0: Level_0, despike: bool = False) -> 'Level_1':
        intensity = lev0.intensity_signal
        if despike:
            intensity_unit = intensity.unit
            warnings.warn('Despiking data, this will take a while ...')
            intensity, mask, stats = img.spikes.identify_and_fix(
                intensity.value, axis=(0, 2, 3),
                percentile_threshold=(0, 99.9),
                poly_deg=1,
            )
            intensity = intensity << intensity_unit
        return cls(
            intensity=intensity,
            start_time=lev0.start_time_signal,
            exposure_length=lev0.requested_exposure_time_signal,
            cam_id=lev0.channel,
            detector=lev0.detector
        )

    def intensity_photons(self, wavelength: u.Quantity) -> u.Quantity:
        return self.detector.dn_to_photon(self.intensity,wavelength)

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
    def from_pickle(cls, path: typ.Optional[pathlib.Path] = None) -> 'Level_1':
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



