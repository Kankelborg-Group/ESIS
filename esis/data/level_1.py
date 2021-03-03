import typing as typ
import dataclasses
import pathlib
import tarfile
import numpy as np
import astropy.io.fits
import astropy.units as u
import kgpy.mixin
import kgpy.obs
import kgpy.img
import kgpy.img.mask
import kgpy.nsroc
import esis.optics
from . import Level_0

__all__ = ['Level_1']


@dataclasses.dataclass
class Level_1(
    kgpy.obs.Image,
    kgpy.mixin.Pickleable,
):
    detector: typ.Optional[esis.optics.Detector] = None
    trajectory: typ.Optional[kgpy.nsroc.Trajectory] = None

    def __post_init__(self):
        super().__post_init__()
        self.update()

    def update(self) -> typ.NoReturn:
        self._despike_result = None

    @classmethod
    def from_level_0(cls, lev0: Level_0) -> 'Level_1':
        return cls(
            intensity=lev0.intensity_signal,
            time=lev0.time_signal,
            exposure_length=lev0.exposure_length_signal,
            channel=lev0.channel,
            time_index=lev0.time_index_signal,
            detector=lev0.detector,
            trajectory=lev0.trajectory,
        )

    def intensity_photons(self, wavelength: u.Quantity) -> u.Quantity:
        return self.detector.convert_electrons_to_photons(self.intensity, wavelength)

    @property
    def despike_result(self):
        if self._despike_result is None:
            intensity, mask, stats = kgpy.img.spikes.identify_and_fix(
                data=self.intensity.value,
                axis=(0, 2, 3),
                percentile_threshold=(0, 99.9),
                poly_deg=1,
            )
            intensity = intensity << self.intensity.unit
            self._despike_result = intensity, mask, stats
        return self._despike_result

    @property
    def intensity_despiked(self):
        return self.despike_result[0]

    @property
    def spikes(self):
        return self.intensity - self.intensity_despiked

    def create_mask(self, sequence):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        import matplotlib.colors as colors

        shp = self.intensity.shape
        poly_list = []
        for i in range(shp[1]):
            img = self.intensity[sequence, i]
            fig, ax = plt.subplots()
            ax.imshow(img, cmap='gray_r', norm=colors.SymLogNorm(1))

            poly = Polygon(kgpy.img.mask.default_vertices(ax), animated=True, facecolor='none')
            ax.add_patch(poly)
            p = kgpy.img.mask.PolygonInteractor(ax, poly)

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

        for sequence in range(self.time.shape[0]):
            for camera in range(self.time.shape[1]):
                name = 'ESIS_Level1_' + str(self.time[sequence, camera]) + '_' + str(camera + 1) + '.fits'
                filename = path / name

                hdr = astropy.io.fits.Header()
                hdr['CAM_ID'] = self.channel[sequence, camera]
                hdr['DATE_OBS'] = str(self.time[sequence, camera])
                hdr['IMG_EXP'] = self.exposure_length[sequence, camera]

                hdul = astropy.io.fits.HDUList()

                hdul.append(astropy.io.fits.PrimaryHDU(np.array(self.intensity[sequence, camera, ...]), hdr))

                hdul.writeto(filename, overwrite=True)

        output_file = path.name + '.tar.gz'

        with tarfile.open(path.parent / output_file, "w:gz") as tar:
            tar.add(path, arcname=path.name)

        return

    def optics_fit(self, guess: esis.optics.Optics):
        pass