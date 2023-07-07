import pathlib
from astropy.io import fits
from astropy import wcs
import astropy.units as u
import typing as typ
import matplotlib.pyplot as plt
import numpy as np
import time
import tarfile
from dataclasses import dataclass
import ndcube
import kgpy.img.coalignment.image_coalignment as img_align
import kgpy.img.mask as img_mask
from esis.data import level_1
import esis
from kgpy.observatories.sdo import aia, hmi
from kgpy.mixin import Pickleable
from kgpy import img

import scipy.optimize
import scipy.ndimage
import scipy.signal

import copy

from matplotlib.patches import Polygon

__all__ = ['ov_Level3_initial',
           'ov_Level3_updated', 'mgx_masks', 'ov_Level3_transforms',
           'ov_Level3_masked', 'hei_transforms', 'ov_final_path', 'hei_final_path', 'ov_final_path_spikes',
           'lev1_despiked']

# intermediate pickles for testing
ov_Level3_initial = pathlib.Path(__file__).parents[1] / 'flight/ov_Level3_initial.pickle'
ov_Level3_updated = pathlib.Path(__file__).parents[1] / 'flight/ov_Level3_updated.pickle'
ov_Level3_transforms = pathlib.Path(__file__).parents[1] / 'flight/ov_Level3_transform.pickle'
ov_Level3_transforms_updated = pathlib.Path(__file__).parents[1] / 'flight/ov_Level3_transform_updated.pickle'
ov_Level3_masked = pathlib.Path(__file__).parents[1] / 'flight/ov_Level3_masked.pickle'
lev1_despiked = pathlib.Path(__file__).parents[1] / 'flight/level_1_despiked.pickle'

mgx_masks = [pathlib.Path(__file__).parents[1] / 'flight/masks/esis_cam{}_mgx_mask.csv'.format(i + 1) for i in range(4)]
hei_masks = [pathlib.Path(__file__).parents[1] / 'flight/masks/esis_cam{}_hei_mask.csv'.format(i + 1) for i in range(4)]

hei_transforms = pathlib.Path(__file__).parents[1] / 'flight/hei_Level3_transform.pickle'
hei_transforms_updated = pathlib.Path(__file__).parents[1] / 'flight/hei_Level3_transform_updated.pickle'

# final pickles
ov_final_path = pathlib.Path(__file__).parents[1] / 'flight/ov_Level3_final.pickle'
ov_final_path_spikes = pathlib.Path(__file__).parents[1] / 'flight/ov_Level3_final_spikes.pickle'
hei_final_path = pathlib.Path(__file__).parents[1] / 'flight/hei_Level3_final.pickle'


@dataclass
class Level_3(Pickleable):
    '''
    The ESIS Level_3 data will be stored in an NDCube.
    The NDCube will contain a 4 axis (time, camera_id, solarx, solary) WCS object.
    '''

    observation: ndcube.NDCube
    transformation_objects: pathlib.Path
    lev1_sequences: np.ndarray
    lev1_cameras: np.ndarray
    spectral_line_name: 'str'
    vignetting_correction_params: np.ndarray = None

    @classmethod
    def from_aia_level1(
            cls,
            lev1: 'level_1.Level_1',
            aia_path: typ.Optional[pathlib.Path] = None,
            line: str = 'ov',

    ) -> 'Level_3':

        """
        Create a Level_3 Obj through a linear co-alignment of ESIS Level1 to AIA 304.

        NOTE!!! This contains hard coded variables that only pertain to the 2019 ESIS Flight, will need to be made
        more general for future launches.  Including a rough FOV and pointing when choosing an AIA cutout should do the
        trick.
        """

        if line != 'ov' or 'hei':
            print('Assuming Spectral Line is O V')
            line = 'ov'

        aia_304 = aia.AIA.from_time_range(
            time_start=lev1.time[0, 0] - 10 * u.s,
            time_end=lev1.time[-1, 0] + 10 * u.s,
            download_path=aia_path,
            channels=[304] * u.AA,
            user_email='jacobdparker@gmail.com'
        )

        cropped_imgs, initial_cropping, lev1 = lev1_prep(lev1, line)

        pad_pix = 400
        initial_pad = ((pad_pix, pad_pix), (pad_pix, pad_pix))
        cropped_imgs = np.pad(cropped_imgs, ((0, 0), (0, 0)) + initial_pad)

        # correct for most of detector tilt and anamorphic distortion
        theta = 16
        scale = 1 / np.cos(np.radians(theta))

        # relevant portion of AIA FOV for ESIS 2019, once we have a rough pointing in Level2 update to grab
        # area around pointing keyword for alignment speed.
        slice1 = slice(1380, 2650)
        slice2 = slice(1435, 2705)
        pos = (slice1, slice2)

        camera = np.array([0, 1, 2, 3])

        sequence = np.arange(cropped_imgs.shape[0])[:-1]

        lev_3_transforms = []
        lev_3_data = np.empty(
            (sequence.shape[0], camera.shape[0], slice1.stop - slice1.start, slice2.stop - slice2.start))

        guess = np.array([[1 / (scale * .79), 1 / .79, 0, 0, 0, 0, -(22.5 + 45 * j)] for j in camera])
        for n, i in enumerate(sequence):
            transform_per_camera = []
            for m, j in enumerate(camera):
                start = time.time()
                print('Fit in Progress: Camera = ', j, ' Sequence = ', i)

                td = aia_304.time - lev1.time[i, 0]  # should be the same for every camera
                best_im = np.abs(td.sec).argmin()
                aia_im = aia_304.intensity[best_im, 0, ...]

                esis_im = cropped_imgs[i, j, ...]

                guess_cam = guess[m]

                bounds = [(guess_cam[0] * .98, guess_cam[0] * 1.02),
                          (guess_cam[1] * .98, guess_cam[1] * 1.02),
                          (0, .01),
                          (0, .01),
                          (0, 1),
                          (0, 1),
                          (-2 + guess_cam[6], 2 + guess_cam[6])]
                # fit = scipy.optimize.differential_evolution(img_align.affine_alignment_quality, bounds,
                #                                             args=(esis_im, aia_im[pos]), workers=4)
                fit = scipy.optimize.minimize(img_align.affine_alignment_quality, guess_cam,
                                              args=(esis_im, aia_im[pos]), bounds=bounds)
                guess[m] = fit.x

                print('Cross-Correlation = ', fit.fun)
                print('Transform = ', fit.x)

                origin = np.array(esis_im.shape) // 2
                esis_im = img_align.modified_affine(esis_im, fit.x, origin)
                # fig,ax = plt.subplots()
                # ax.imshow(esis_im, vmax = np.percentile(esis_im,99.99))
                # plt.show()

                aia_cc = np.empty_like(aia_im)
                aia_cc[pos] = aia_im[pos]
                cc = img_align.normalized_cc(aia_cc, esis_im)

                trans = np.unravel_index(cc.argmax(), cc.shape)
                trans = (-esis_im.shape[0] // 2 + trans[0], -esis_im.shape[1] // 2 + trans[1])

                aia_shp = aia_im.shape
                big_esis = np.empty(aia_shp)
                big_esis[0:esis_im.shape[0], 0:esis_im.shape[1]] = esis_im

                # move based on cc
                big_esis = np.roll(big_esis, trans, (0, 1))

                lev_3_data[n, m, ...] = big_esis[pos]
                # fix,ax = plt.subplots()
                # ax.imshow(lev_3_data[n,m,...], vmin = 0, vmax = np.percentile(lev_3_data[n,m,...],99))

                transform_per_camera.append(img_align.ImageTransform(fit.x, origin, img_align.modified_affine,
                                                                     initial_cropping, initial_pad, aia_shp, trans,
                                                                     pos))
                print('Fit Duration = ', time.time() - start)
            lev_3_transforms.append(transform_per_camera)

        # plt.show()
        aia_wcs = aia_304.wcs[0, 0].slice(pos)
        date_obs = lev1.time[sequence[0], 0]
        time_delta = lev1.time[sequence[1], 0] - date_obs

        # Axis 3 and 4 of the WCS object are camera and sequence respectively to match the Level1 ndarray
        # For future runs of ESIS with additional cameras CRVAL3 will require modification.
        lev_3_header = dict([('NAXIS1', aia_wcs._naxis[0]), ('NAXIS2', aia_wcs._naxis[1]), ('NAXIS3', camera.shape[0]),
                             ('NAXIS4', sequence.shape[0]),
                             ('DATEOBS', str(date_obs)), ('DATEREF', str(date_obs)), ('MJDREF', date_obs.mjd),
                             ('CTYPE1', aia_wcs.wcs.ctype[0]), ('CTYPE2', aia_wcs.wcs.ctype[1]),
                             ('CTYPE3', 'CAMERA_ID'), ('CTYPE4', 'UTC'),
                             ('CRVAL1', aia_wcs.wcs.crval[0]), ('CRVAL2', aia_wcs.wcs.crval[1]), ('CRVAL3', 1),
                             ('CRVAL4', time_delta.sec),
                             ('CRPIX1', aia_wcs.wcs.crpix[0]), ('CRPIX2', aia_wcs.wcs.crpix[1]), ('CRPIX3', 0),
                             ('CRPIX4', 1),
                             ('CUNIT1', str(aia_wcs.wcs.cunit[0])), ('CUNIT2', str(aia_wcs.wcs.cunit[1])),
                             ('CUNIT3', 'pix'), ('CUNIT4', 's'),
                             ('CDELT1', aia_wcs.wcs.cdelt[0]), ('CDELT2', aia_wcs.wcs.cdelt[1]), ('CDELT3', 1),
                             ('CDELT4', time_delta.sec),
                             ])
        lev_3_wcs = wcs.WCS(lev_3_header)

        meta = dict([("Description", "Level_3 was formed via a linear co-alignment of ESIS Level-1 and AIA 304"),
                     ])
        lev_3_ndcube = ndcube.NDCube(lev_3_data, lev_3_wcs, meta=meta)

        lev_3_transform_cube = img_align.TransformCube(lev_3_transforms)
        if line == 'ov':
            lev_3_transform_cube.to_pickle(ov_Level3_transforms)
            transform_path = ov_Level3_transforms
        if line == 'hei':
            lev_3_transform_cube.to_pickle(hei_transforms)
            transform_path = hei_transforms

        return cls(observation=lev_3_ndcube, transformation_objects=transform_path,
                   lev1_sequences=sequence, lev1_cameras=camera, spectral_line_name=line)

    def update_internal_alignment(self, lev1, ref_channel=1) -> 'Level_3':

        line = self.spectral_line_name
        cropped_imgs, initial_cropping, lev1 = lev1_prep(lev1, line)
        lev3_transform_cube = img_align.TransformCube.from_pickle(self.transformation_objects)

        for lev3_seq, seq in enumerate(self.lev1_sequences):
            for lev3_cam, cam in enumerate(self.lev1_cameras):

                ref_img = self.observation.data[lev3_seq, ref_channel, ...]
                fit_img = lev1.intensity[seq, cam, ...]
                img_transform = lev3_transform_cube.transform_cube[lev3_seq][lev3_cam]

                guess = img_align.modified_affine_to_quadratic(img_transform.transform, img_transform.origin)

                t = 2e-5
                bounds = [(-1, 1), (-1, 1), (-1, 1), (-t, t), (-t, t), (-t, t),
                          (-1, 1), (-1, 1), (-1, 1), (-t, t), (-t, t), (-t, t)]
                img_transform.transform = guess
                img_transform.transform_func = img_align.quadratic_transform

                # additional logic required to skip fit of ref channel while still updating the transform object
                if cam != ref_channel:
                    start = time.time()
                    print('Fit in Progress: Camera = ', cam, ' Sequence = ', seq)
                    # fit = scipy.optimize.differential_evolution(img_align.affine_alignment_quality, bounds,
                    #                                             args=(fit_img, ref_img), workers=-1)
                    fit = scipy.optimize.minimize(img_align.test_alignment_quality, guess, bounds=bounds,
                                                  args=(fit_img, ref_img, img_transform))

                    print('Fit Duration = ', time.time() - start)
                    print('Cross-Correlation = ', fit.fun)
                    print('Transform = ', fit.x)

                    img_transform.transform = fit.x

                    fit_img = img_transform.transform_image(fit_img)
                    # fig, ax = plt.subplots()
                    # ax.imshow(fit_img)
                    # plt.show()
                    self.observation.data[lev3_seq, lev3_cam] = fit_img

                lev3_transform_cube.transform_cube[lev3_seq][lev3_cam] = img_transform
        if line == 'ov':
            lev3_transform_cube.to_pickle(ov_Level3_transforms_updated)
            self.transformation_objects = ov_Level3_transforms_updated
        if line == 'hei':
            lev3_transform_cube.to_pickle(hei_transforms_updated)
            self.transformation_objects = hei_transforms_updated
        self.observation.meta[
            'Description'] = 'Level 3 data produced via coalignment with AIA 304 and an updated internal ' \
                             'alignment of ESIS channels to Channel Index ' + str(ref_channel)

        return self

    def add_mask(self, lev1) -> 'Level_3':
        '''
        Transform masks created for Level1 data into Level_3 coordinates and add to Level_3 NDCube

        This can possibly be done better by just overwriting the tranform and applying transform coords???
        '''

        line = self.spectral_line_name
        if line == 'ov':
            mask_coords = [np.genfromtxt(path, delimiter=',') for path in mgx_masks]
        if line == 'hei':
            mask_coords = [np.genfromtxt(path, delimiter=',') for path in hei_masks]
        mask_cube = np.empty_like(self.observation.data)
        transforms = img_align.TransformCube.from_pickle(self.transformation_objects)
        for l3_seq, l1_seq in enumerate(self.lev1_sequences):
            for l3_cam, l1_cam in enumerate(self.lev1_cameras):
                m_coord = np.expand_dims(mask_coords[l1_cam].T, axis=-1)
                l3_img = self.observation.data[l3_seq, l3_cam]

                l1_img = lev1.intensity[l1_seq, l1_cam]
                transform_obj = transforms.transform_cube[l3_seq][l3_cam]

                forward_mask_transform = transform_obj.invert_quadratic_transform(l1_img)

                m_coord = transform_obj.coord_pre_process(m_coord)
                m_coord = img_align.quadratic_transform(l1_img, forward_mask_transform, transform_obj.origin,
                                                        old_coord=m_coord[::-1, :, :])

                m_coord = m_coord[::-1, :,
                          :]  # required flip and flip back of the mask coordinate likely do to patch xy and image row column being different
                # print(np.squeeze(m_coord).T)

                m_coord = transform_obj.coord_post_process(m_coord)

                poly = Polygon(np.squeeze(m_coord).T)
                # fig, ax = plt.subplots()
                # ax.imshow(l3_img)
                # ax.add_patch(poly)
                # plt.show()

                mask_cube[l3_seq, l3_cam] = img_mask.make_mask(poly, l3_img.shape)

        self.observation.mask = mask_cube

        return self

    def correct_vignetting(self, vignetting_params=None):
        if self.vignetting_correction_params is None and vignetting_params is None:
            self.vignetting_correction_params = self.find_vignetting_correction()
            self.observation.data[...] /= self.vignetting_correction()
        elif self.vignetting_correction_params is not None and vignetting_params is not None:
            print('Vignetting Already Corrected')
        elif self.vignetting_correction_params is not None and vignetting_params is None:
            print('Vignetting Already Corrected')
        elif self.vignetting_correction_params is None and vignetting_params is not None:
            self.vignetting_correction_params = vignetting_params
            self.observation.data[...] /= self.vignetting_correction()

        return self

    def vignetting_correction(self, scale_factor=None) -> np.ndarray:

        if scale_factor is None:
            if self.vignetting_correction_params is None:
                print('Corect Vignetting First')
                assert False
            else:
                scale_factor = self.vignetting_correction_params

        octagon_size_pix = 1140
        octagon_edge_pix = 65

        img_shp = self.observation.data.shape
        x, y = img_align.get_img_coords(self.observation.data[0, 0])
        vignette_correction = np.empty_like(self.observation.data)

        # pointing drift calculated manually from level3 outputs, could be possible to incoorperate actual mapping to do this better
        x_drift = (1265 - 1257) / img_shp[0]
        y_drift = (632 - 636) / img_shp[0]

        x0, y0 = np.array(img_shp[2:]) // 2

        for l3_cam, l1_cam in enumerate(self.lev1_cameras):
            for l3_seq, l1_seq in enumerate(self.lev1_sequences):
                rot_angle = 180 - 22.5 - 45 * l1_cam

                c = np.cos(np.deg2rad(rot_angle))
                s = np.sin(np.deg2rad(rot_angle))

                x_ = (c * (x - x0 - x_drift * l3_seq) - s * (y - y0 - y_drift * l3_seq)) + x0

                rot_z = scale_factor[l3_cam] / octagon_size_pix * (x_ - octagon_edge_pix) + 1

                vignette_correction[l3_seq, l3_cam] = rot_z

        return vignette_correction

    def find_vignetting_correction(self):

        guess = np.array([.4, .4, .4, .4])
        bounds = [(.2, .8), (.2, .8), (.2, .8), (.2, .8)]
        start = time.time()
        print('Finding Vignetting Correction')
        # fit = scipy.optimize.minimize(vignetting_correction_quality, guess, args=(self,), bounds=bounds,
        #                               options={'ftol': 1e-5})
        # fit = scipy.optimize.differential_evolution(vignetting_correction_quality, bounds, args=(self,), polish=True, workers=4)
        fit = scipy.optimize.differential_evolution(vignetting_correction_quality, bounds, args=(self,), polish=True,
                                                    workers=-1)
        print('Fit Duration = ', time.time() - start)
        print('Fit Params = ', fit.x)
        return fit.x

    @classmethod
    def from_pickle(cls, path: typ.Optional[pathlib.Path] = None) -> 'Level_3':
        obs = super().from_pickle(path)
        obs.observation.wcs.array_shape = obs.observation.data.shape
        return obs

    def to_fits(self, path: pathlib.Path, label='ESIS_Level3_'):
        '''
        In need of a rework since moving to NDCube.  Note that WCS.to_header does not output naxis keywords
        '''

        path.mkdir(parents=True, exist_ok=True)
        # len1 = self.observation.data.shape[0]
        # len2 = self.observation.data.shape[1]
        #
        # date_obs = self.time
        # for sequence in range(len1):
        #     for camera in range(len2):
        #         data_name = label + str(sequence) + '_' + date_obs[sequence] + '_' + str(camera + 1) + '.fits'
        #         data_filename = path / data_name
        #
        #         mask_name = label + str(sequence) + '_' + date_obs[sequence] + '_' + str(camera + 1) + '_mask.fits'
        #         mask_filename = path / mask_name
        #
        #         hdr = self.observation[sequence, camera].wcs.dropaxis(-1).dropaxis(-1).to_header()
        #         # hdr['CAM_ID'] = self.cam_id[sequence, camera]
        #
        #         hdr['DATE_OBS'] = date_obs
        #
        #         hdul = fits.HDUList()
        #         hdul.append(fits.PrimaryHDU(np.array(self.observation.data[sequence, camera, ...]), hdr))
        #         hdul.writeto(data_filename, overwrite=True)
        #
        #         hdul_mask = fits.HDUList()
        #         hdul_mask.append(fits.PrimaryHDU(np.array(self.observation.mask[sequence, camera, ...]), hdr))
        #         hdul.writeto(data_filename, overwrite=True)

        hdr = self.observation.wcs.to_header()

        hdul = fits.HDUList()
        hdul.append(fits.PrimaryHDU(self.observation.data, hdr))
        data_path = path / 'ESIS_level3.fits'
        hdul.writeto(data_path)

        output_file = data_path.name + '.tar.gz'
        with tarfile.open(path / output_file, "w:gz") as tar:
            tar.add(data_path, arcname=data_path.name)

        hdul = fits.HDUList()
        hdul.append(fits.PrimaryHDU(self.observation.mask, hdr))
        mask_path = path / 'ESIS_level3_mgx_mask.fits'
        hdul.writeto(mask_path)

        output_file = mask_path.name + '.tar.gz'
        with tarfile.open(path / output_file, "w:gz") as tar:
            tar.add(mask_path, arcname=mask_path.name)



    def to_aia_object(self, aia_channel=304 * u.AA) -> 'Level_3':
        '''
        Replace all images in a Level 3 object with co-temporal AIA images.
        '''
        aia_obj = copy.deepcopy(self)
        times = self.time
        aia_304 = aia.AIA.from_time_range(times[0] - 20 * u.s, times[-1] + 20 * u.s, channels=[aia_channel],
                                          user_email='jacobdparker@gmail.com')

        transforms = img_align.TransformCube.from_pickle(self.transformation_objects)
        aia_times = []
        for l3_seq, l1_seq in enumerate(self.lev1_sequences):
            td = aia_304.time - times[l3_seq]  # should be the same for every camera
            best_im = np.abs(td.sec).argmin()
            aia_im = aia_304.intensity[best_im, 0]
            aia_times.append(aia_304.time[best_im, 0])
            for l3_cam, l1_cam in enumerate(self.lev1_cameras):
                crop = transforms.transform_cube[l3_seq][l3_cam].post_transform_crop
                aia_obj.observation.data[l3_seq, l3_cam] = aia_im[crop]
        aia_obj.observation.meta['times'] = aia_times
        return aia_obj

    def to_hmi_object(self) -> 'Level_3':
        '''
        Replace all images in a Level 3 object with co-temporal AIA images.
        '''
        hmi_obj = copy.deepcopy(self)
        times = self.time
        hmi_data = hmi.HMI.from_time_range(times[0] - 20 * u.s, times[-1] + 20 * u.s,
                                           user_email='jacobdparker@gmail.com')

        transforms = img_align.TransformCube.from_pickle(self.transformation_objects)
        hmi_times = []
        for l3_seq, l1_seq in enumerate(self.lev1_sequences):
            td = hmi_data.time - times[l3_seq]  # should be the same for every camera
            best_im = np.abs(td.sec).argmin()
            hmi_im = hmi_data.intensity[best_im, 0]
            hmi_times.append(hmi_data.time[best_im, 0])
            for l3_cam, l1_cam in enumerate(self.lev1_cameras):
                crop = transforms.transform_cube[l3_seq][l3_cam].post_transform_crop
                hmi_obj.observation.data[l3_seq, l3_cam] = hmi_im[crop]
        hmi_obj.observation.meta['times'] = hmi_times
        return hmi_obj

    def normalize_intensities(self):
        '''
        Normalizes all channels to the highest mean (normalizing interchannel intensity and correcting for absorption)
        Returns
        -------
        self

        '''

        if self.vignetting_correction_params is None:
            print('Best to correct vignetting first, proceeding anyway')

        means = self.masked_means
        highest_mean = means.max(axis=(0, 1), keepdims=True)

        self.observation.data[...] *= highest_mean / means

        return self

    @property
    def masked_means(self) -> np.ndarray:
        '''
        Given a masked level3 object this routine will return the mean taken from the portion of the sun seen by all 4
        cameras that does not contain the bright MgX line as a cube by which level3.observation.data can be divided.

        For best results correct vignetting first.
        '''

        super_mask = np.ones_like(self.observation.mask[:, 0])
        for i, cam in enumerate(self.lev1_cameras):
            super_mask *= self.observation.mask[:, i]

        super_mask = np.resize(super_mask[:, None], self.observation.mask.shape)
        super_mask = np.resize(super_mask[:, None], self.observation.mask.shape)
        masked_cube = self.observation.data * super_mask
        masked_cube[masked_cube == 0] = np.nan
        mean_cube = np.nanmean(masked_cube, axis=(-2, -1))

        return mean_cube[..., None, None]

    @property
    def time(self):
        shp = self.observation.data.shape[0]
        time_pix = np.arange(shp)
        pix = np.zeros(shp)
        times = self.observation.wcs.pixel_to_world(pix, pix, pix, time_pix)[-1]
        times.format = 'isot'
        return times

    @property
    def min_images(self):
        return np.min(self.observation.data, axis=1, keepdims=True)


def vignetting_correction_quality(x, lev3: 'Level_3'):
    scale_factor = x
    vignetting = lev3.vignetting_correction(scale_factor)

    camera_combos = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    # channel_index = [[0,1],[0,2],[1,2]]

    sequences = [(i, j) for (i, j) in enumerate(lev3.lev1_sequences)]
    slope = []
    for lev3_seq, lev1_seq in sequences[::1]:

        mask1 = lev3.observation.mask[lev3_seq, 0]
        mask2 = lev3.observation.mask[lev3_seq, 1]
        mask3 = lev3.observation.mask[lev3_seq, 2]
        mask4 = lev3.observation.mask[lev3_seq, 3]
        masks = [mask1, mask2, mask3, mask4]
        super_mask = mask1 * mask2 * mask3 * mask4

        im1 = lev3.observation.data[lev3_seq, 0] / vignetting[lev3_seq, 0]
        im1 /= np.mean(im1[im1 * super_mask != 0])
        im2 = lev3.observation.data[lev3_seq, 1] / vignetting[lev3_seq, 1]
        im2 /= np.mean(im2[im2 * super_mask != 0])
        im3 = lev3.observation.data[lev3_seq, 2] / vignetting[lev3_seq, 2]
        im3 /= np.mean(im3[im3 * super_mask != 0])
        im4 = lev3.observation.data[lev3_seq, 3] / vignetting[lev3_seq, 3]
        im4 /= np.mean(im4[im4 * super_mask != 0])

        normalized_images = np.array([im1, im2, im3, im4])

        for in1, in2 in camera_combos:
            dif_im = normalized_images[in1] - normalized_images[in2]
            # fig,ax = plt.subplots()
            # ax.imshow(dif_im,vmin = -2,vmax = 2)
            # plt.show()

            # dif_im *= super_mask
            dif_im *= masks[in1] * masks[in2]
            dif_im[dif_im == 0] = np.nan

            column_mean = np.nanmean(dif_im, axis=0)
            column_mean = column_mean[300:-300]
            # column_mean = column_mean[column_mean != 0]
            # fig,ax = plt.subplots()
            # ax.plot(column_mean)
            # plt.show()

            poly_fit = np.polynomial.Polynomial.fit(np.arange(column_mean.shape[0]), column_mean, deg=1)
            fit_slope = poly_fit.coef[1]
            slope.append(fit_slope)

    slope = np.array(slope)
    # slope_max = np.array(slope).max()
    # slope_total = np.sum(np.array(np.abs(slope)))
    least_squares = np.sqrt(np.sum(np.square(slope))) / slope.size
    print(least_squares)
    return least_squares


def lev1_prep(lev1, line=None):
    l1 = copy.deepcopy(lev1)

    if line == 'hei':
        l1.intensity = l1.intensity_photons(584 * u.AA)
        initial_cropping = (slice(None), slice(l1.intensity.shape[-1] // 2 - 225))

    if line == 'ov':
        l1.intensity = l1.intensity_photons(630 * u.AA)
        initial_cropping = (slice(None), slice(l1.intensity.shape[-1] // 2 - 25, None))

    # undo flip about short axis from optical system
    l1.intensity = np.flip(l1.intensity, axis=-2)
    cropped_imgs = l1.intensity[(slice(None), slice(None)) + initial_cropping]
    return cropped_imgs, initial_cropping, l1


def full_level3_prep(despike=True, line=None, full_prep=False):
    lev1 = esis.flight.level_1()
    if despike:
        if lev1_despiked.exists() and not full_prep:
            print('Loading Despiked Level1 Data')
            lev1 = level_1.Level_1.from_pickle(lev1_despiked)
        else:
            intensity_unit = lev1.intensity.unit
            print('Despiking L1 data, this will take a while ...')
            intensity, mask, stats = img.spikes.identify_and_fix(
                data=lev1.intensity.value,
                axis=(0, 2, 3),
                percentile_threshold=(0, 99.9),
                poly_deg=1,
            )
            lev1.intensity = intensity << intensity_unit
            lev1.to_pickle(lev1_despiked)

    if line == 'ov' or line is None:
        if ov_Level3_initial.exists() and not full_prep:
            print('Loading' + str(ov_Level3_initial))
            lev3_initial = Level_3.from_pickle(ov_Level3_initial)
        else:
            lev3_initial = Level_3.from_aia_level1(lev1, line=line)
            lev3_initial.to_pickle(ov_Level3_initial)

        if ov_Level3_updated.exists() and not full_prep:
            print('Loading' + str(ov_Level3_updated))
            lev3_updated = Level_3.from_pickle(ov_Level3_updated)
        else:
            lev3_updated = lev3_initial.update_internal_alignment(lev1)
            lev3_updated.to_pickle(ov_Level3_updated)

        if ov_Level3_masked.exists() and not full_prep:
            print('Loading' + str(ov_Level3_masked))
            lev3_masked = Level_3.from_pickle(ov_Level3_masked)
        else:
            lev3_masked = lev3_updated.add_mask(lev1)
            lev3_updated.to_pickle(ov_Level3_masked)

        lev3_final = lev3_masked.correct_vignetting()
        lev3_final = lev3_final.normalize_intensities()

        if despike:
            lev3_final.to_pickle(ov_final_path)
        else:
            lev3_final.to_pickle(ov_final_path_spikes)

    return lev3_final
