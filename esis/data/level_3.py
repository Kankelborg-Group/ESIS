import pathlib
import pickle
from astropy.io import fits
from astropy import wcs
import astropy.units as u
import typing as typ

import numpy as np
import time
import tarfile
from dataclasses import dataclass
import ndcube
import kgpy.img.coalignment.image_coalignment as img_align
import kgpy.img.masks.mask as img_mask
from esis.data import level_1
from kgpy.observatories.aia import aia
from kgpy.mixin import Pickleable

import scipy.optimize
import scipy.ndimage
import scipy.signal

from matplotlib.patches import Polygon

__all__ = ['ov_Level3_initial', 'default_aia_path', 'ov_Level3_updated', 'mgx_masks', 'ov_Level3_transforms',
           'ov_Level3_masked', 'hei_transforms', 'ov_final_path', 'hei_final_path']

default_aia_path = pathlib.Path(aia.__file__).parent / 'data/'

# intermediate pickles for testing
ov_Level3_initial = pathlib.Path(__file__).parent / 'ov_Level3.pickle'
ov_Level3_updated = pathlib.Path(__file__).parent / 'ov_Level3_updated.pickle'
ov_Level3_transforms = pathlib.Path(__file__).parent / 'esis_Level3_transform.pickle'
ov_Level3_transforms_updated = pathlib.Path(__file__).parent / 'esis_Level3_transform_updated.pickle'
ov_Level3_masked = pathlib.Path(__file__).parent / 'ov_Level3_masked.pickle'

mgx_masks = [pathlib.Path(__file__).parent / 'masks/esis_cam{}_mgx_mask.csv'.format(i+1) for i in range(4)]
hei_masks = [pathlib.Path(__file__).parent / 'masks/esis_cam{}_hei_mask.csv'.format(i + 1) for i in range(4)]

hei_transforms = pathlib.Path(__file__).parent / 'heI_Level3_transform.pickle'
hei_transforms_updated = pathlib.Path(__file__).parent / 'heI_Level3_transform_updated.pickle'

# final pickles
ov_final_path = pathlib.Path(__file__).parent / 'ov_Level3_final.pickle'
hei_final_path = pathlib.Path(__file__).parent / 'hei_Level3_final.pickle'


@dataclass
class Level3(Pickleable):
    '''
    The ESIS Level3 data will be stored in an NDCube.
    The NDCube will contain a 4 axis (time, camera_id, solarx, solary) WCS object.
    '''

    observation: ndcube.NDCube
    transformation_objects: pathlib.Path
    lev1_sequences: np.ndarray
    lev1_cameras: np.ndarray


    @classmethod
    def from_aia_level1(cls, aia_path: pathlib.Path = default_aia_path,
                        level1_path: pathlib.Path = level_1.Level1.default_pickle_path(), hei = False) -> 'Level3':

        """
        Create a Level3 Obj through a linear co-alignment of ESIS Level1 to AIA 304.

        NOTE!!! This contains hard coded variables that only pertain to the 2019 ESIS Flight, will need to be made
        more general for future launches.  Including a rough FOV and pointing when choosing an AIA cutout should do the
        trick.
        """

        esis = level_1.Level1.from_pickle(level1_path)

        aia_channel = [304*u.angstrom]
        print(esis.start_time.shape)
        start_time = esis.start_time[0,0]
        end_time = esis.start_time[-1,0]
        print(start_time)
        print(end_time)

        aia_304_files = aia.fetch_from_time(start_time, end_time, default_aia_path, aia_channels=aia_channel)
        aia_304_lev15 = aia.aiaprep_from_paths(aia_304_files)

        aia_304 = aia.AIA.from_path('aia_304', aia_304_lev15)

        if hei == False:
            inital_cropping = (slice(None),slice(esis.intensity.shape[-1] // 2 - 25, None))
        else:
            inital_cropping = (slice(None), slice(esis.intensity.shape[-1] // 2-225))

        cropped_imgs = esis.intensity[(slice(None), slice(None)) + inital_cropping]

        pad_pix = 400
        initial_pad = ((pad_pix, pad_pix), (pad_pix, pad_pix))
        cropped_imgs = np.pad(cropped_imgs, ((0, 0), (0, 0))+ initial_pad)



        # correct for most of detector tilt and anamorphic distortion
        theta = 16
        scale = 1 / np.cos(np.radians(theta))
        
        #relevant portion of AIA FOV for ESIS 2019
        slice1 = slice(1380, 2650)
        slice2 = slice(1435, 2705)
        pos = (slice1, slice2)

        camera = np.array([0, 1, 2, 3])
        # camera = np.array([3])
        sequence = np.arange(cropped_imgs.shape[0])[4:-3]
        # sequence = np.array([15,16])


        lev_3_transforms = []
        # lev_3_transforms = np.empty((sequence.shape[0], camera.shape[0]),dtype = image_coalignment.ImageTransform)
        lev_3_data = np.empty(
            (sequence.shape[0], camera.shape[0], slice1.stop - slice1.start, slice2.stop - slice2.start))

        guess = np.array([[1 / (scale * .79), 1 / .79, 0, 0, 0, 0, -(22.5 + 45 * j)] for j in camera])
        for n, i in enumerate(sequence):
            transform_per_camera = []
            for m, j in enumerate(camera):

                start = time.time()
                print('Fit in Progress: Camera = ',j,' Sequence = ',i)

                td = aia_304.exposure_start_time - esis.start_time[i, 0]  # should be the same for every camera
                best_im = np.abs(td.sec).argmin()
                aia_im = aia_304.intensity[best_im, ...]

                esis_im = cropped_imgs[i, j, ...]

                guess_cam = guess[j]
                
                bounds = [(guess_cam[0] * .98, guess_cam[0] * 1.02),
                          (guess_cam[1] * .98, guess_cam[1] * 1.02),
                          (0, .01),
                          (0, .01),
                          (0, 1),
                          (0, 1),
                          (-2 + guess_cam[6], 2 + guess_cam[6])]
                # fit = scipy.optimize.differential_evolution(img_align.affine_alignment_quality,bounds,args =(esis_im, aia_im[pos]),workers = -1)
                fit = scipy.optimize.minimize(img_align.affine_alignment_quality, guess_cam, args = (esis_im, aia_im[pos]), bounds = bounds)
                guess[j] = fit.x

                print('Cross-Correlation = ', fit.fun)
                print('Transform = ', fit.x)

                origin = np.array(esis_im.shape) // 2
                esis_im = img_align.modified_affine(esis_im, fit.x, origin)
                # fig,ax = plt.subplots()
                # ax.imshow(esis_im)
                # plt.show()



                aia_cc = np.empty_like(aia_im)
                aia_cc[pos] = aia_im[pos]
                cc = scipy.signal.correlate((aia_cc - np.mean(aia_cc[pos])) / np.std(aia_cc[pos]),
                                            (esis_im - np.mean(esis_im)) / np.std(esis_im), mode='same')



                trans = np.unravel_index(cc.argmax(), cc.shape)
                trans = (-esis_im.shape[0] // 2 + trans[0], -esis_im.shape[1] // 2 + trans[1])

                aia_shp = aia_im.shape
                big_esis = np.empty(aia_shp)
                big_esis[0:esis_im.shape[0], 0:esis_im.shape[1]] = esis_im

                # move based on cc
                big_esis = np.roll(big_esis, trans, (0, 1))

                lev_3_data[n,m,...] = big_esis[pos]
                # fix,ax = plt.subplots()
                # ax.imshow(lev_3_data[n,m,...])
                # plt.show()

                transform_per_camera.append(img_align.ImageTransform(fit.x, origin, img_align.modified_affine,
                                                                     inital_cropping, initial_pad, aia_shp, trans, pos))
                print('Fit Duration = ', time.time()-start)
            lev_3_transforms.append(transform_per_camera)
        aia_wcs = aia_304.wcs[0].slice(pos)
        date_obs = esis.start_time[sequence[0], 0]
        time_delta = esis.start_time[sequence[1], 0] - date_obs

        # Axis 3 and 4 of the WCS object are camera and sequence respectively to match the Level1 ndarray
        # For future runs of ESIS with additional cameras CRVAL3 will require modification.
        lev_3_header = dict([('NAXIS1',aia_wcs._naxis[0]), ('NAXIS2',aia_wcs._naxis[1]), ('NAXIS3', camera.shape[0]), ('NAXIS4',sequence.shape[0]),
                             ('DATEOBS', str(date_obs)), ('DATEREF', str(date_obs)), ('MJDREF', date_obs.mjd),
                             ('CTYPE1',aia_wcs.wcs.ctype[0]), ('CTYPE2',aia_wcs.wcs.ctype[1]),('CTYPE3', 'CAMERA_ID'), ('CTYPE4','UTC'),
                             ('CRVAL1',aia_wcs.wcs.crval[0]),('CRVAL2',aia_wcs.wcs.crval[1]),('CRVAL3',1),('CRVAL4',time_delta.sec),
                             ('CRPIX1',aia_wcs.wcs.crpix[0]),('CRPIX2',aia_wcs.wcs.crpix[1]),('CRPIX3', 0), ('CRPIX4', 1),
                             ('CUNIT1',str(aia_wcs.wcs.cunit[0])),('CUNIT2',str(aia_wcs.wcs.cunit[1])), ('CUNIT3','pix'),('CUNIT4','s'),
                             ('CDELT1',aia_wcs.wcs.cdelt[0]),('CDELT2', aia_wcs.wcs.cdelt[1]), ('CDELT3',1),('CDELT4', time_delta.sec),
                             ])
        lev_3_wcs = wcs.WCS(lev_3_header)

        meta = dict([("Description", "Level3 was formed via a co-alignment of ESIS level1 and AIA 304"),
                     ])
        lev_3_ndcube = ndcube.NDCube(lev_3_data,lev_3_wcs,meta = meta)




        lev_3_transform_cube = img_align.TransformCube(lev_3_transforms)
        if hei == False:
            lev_3_transform_cube.to_pickle(ov_Level3_transforms)
            transform_path = ov_Level3_transforms
        else:
            lev_3_transform_cube.to_pickle(hei_transforms)
            transform_path = hei_transforms


        return cls(observation=lev_3_ndcube,transformation_objects=transform_path,
                   lev1_sequences=sequence,lev1_cameras=camera)

    def update_internal_alignment(self,ref_channel = 1, heI = False) -> 'Level3':
        lev1 = level_1.Level1.from_pickle(level_1.default_pickle_path)
        print(self.transformation_objects)
        lev1_transforms = img_align.TransformCube.from_pickle(self.transformation_objects)


        for lev3_seq, seq in enumerate(self.lev1_sequences):
            for lev3_cam, cam in enumerate(self.lev1_cameras):

                ref_img = self.observation.data[lev3_seq,ref_channel,...]
                fit_img = lev1.intensity[seq,cam,...]
                img_transform = lev1_transforms.transform_cube[lev3_seq][lev3_cam]

                guess = img_align.modified_affine_to_quadratic(img_transform.transform,img_transform.origin)

                t = 2e-5
                bounds = [(-1,1),(-1,1),(-1,1),(-t,t),(-t,t),(-t,t),
                          (-1,1),(-1,1),(-1,1),(-t,t),(-t,t),(-t,t)]
                img_transform.transform = guess
                img_transform.transform_func = img_align.quadratic_transform


                # additional logic required to skip fit of ref channel while still updating the transform object
                if cam != ref_channel:
                    start = time.time()
                    print('Fit in Progress: Camera = ', cam, ' Sequence = ', seq)
                    # options = dict([('gtol', .1)])
                    fit = scipy.optimize.minimize(img_align.test_alignment_quality, guess,bounds = bounds,
                                                args = (fit_img, ref_img, img_transform ))

                    print('Fit Duration = ', time.time() - start)
                    print('Cross-Correlation = ', fit.fun)
                    print('Transform = ', fit.x)

                    img_transform.transform = fit.x

                    fit_img  = img_transform.transform_image(fit_img)
                    # fig, ax = plt.subplots()
                    # ax.imshow(fit_img)
                    # plt.show()
                    self.observation.data[lev3_seq,lev3_cam] = fit_img

                lev1_transforms.transform_cube[lev3_seq][lev3_cam] = img_transform
        if heI == False:
            lev1_transforms.to_pickle(ov_Level3_transforms_updated)
            self.transformation_objects = ov_Level3_transforms_updated
        else:
            lev1_transforms.to_pickle(hei_transforms_updated)
            self.transformation_objects = hei_transforms_updated
        self.observation.meta['Description'] = 'Level 3 data produced via coalignment with AIA 304 and an updated internal ' \
                                              'alignment of ESIS channels'

        return self

    def add_mask(self, line = None) -> 'Level3':
        '''
        Transform masks created for Level1 data into Level3 coordinates and add to Level3 NDCube
        '''
        lev1 = level_1.Level1.from_pickle()
        if line == 'mgx':
            mask_coords = [np.genfromtxt(path,delimiter=',') for path in mgx_masks]
        if line == 'hei':
            mask_coords = [np.genfromtxt(path, delimiter=',') for path in hei_masks]
        if line == None:
            print('Please Select Spectral Line')
        mask_cube = np.empty_like(self.observation.data)
        transforms = img_align.TransformCube.from_pickle(self.transformation_objects)
        for l3_seq, l1_seq in enumerate(self.lev1_sequences):
            for l3_cam,l1_cam in enumerate(self.lev1_cameras):
                m_coord = np.expand_dims(mask_coords[l1_cam].T,axis=-1)
                l3_img = self.observation.data[l3_seq,l3_cam]

                l1_img = lev1.intensity[l1_seq,l1_cam]
                transform_obj = transforms.transform_cube[l3_seq][l3_cam]

                forward_mask_transform = transform_obj.invert_quadratic_transform(l1_img)

                m_coord = transform_obj.coord_pre_process(m_coord)
                m_coord = img_align.quadratic_transform(l1_img,forward_mask_transform,transform_obj.origin,old_coord = m_coord[::-1,:,:])

                m_coord = m_coord[::-1,:,:]  #required flip and flip back of the mask coordinate likely do to patch xy and image row column being different
                # print(np.squeeze(m_coord).T)

                m_coord = transform_obj.coord_post_process(m_coord)

                poly = Polygon(np.squeeze(m_coord).T)
                # fig, ax = plt.subplots()
                # ax.imshow(l3_img)
                # ax.add_patch(poly)
                # plt.show()


                mask_cube[l3_seq,l3_cam] = img_mask.make_mask(poly,l3_img.shape)

        self.observation.mask = mask_cube

        return

    def correct_vignetting(self,scale_factor,fudge_angle = np.array([0,0,0,0])) -> np.ndarray:

        octagon_size_pix = 1140
        octagon_edge_pix = 65

        img_shp = self.observation.data.shape
        x,y = img_align.get_img_coords(self.observation.data[0,0])
        vignette_correction = np.empty_like(self.observation.data)

        # pointing drift calculated manually from level3 outputs, could be possible to incoorperate actual mapping to do this better
        x_drift = (1265 - 1257) / img_shp[0]
        y_drift = (632 - 636) / img_shp[0]

        x0,y0 = np.array(img_shp[2:]) // 2


        for l3_cam,l1_cam in enumerate(self.lev1_cameras):
            for l3_seq,l1_seq in enumerate(self.lev1_sequences):

                rot_angle =180 -22.5 - 45 * l1_cam + fudge_angle[l3_cam]

                c = np.cos(np.deg2rad(rot_angle))
                s = np.sin(np.deg2rad(rot_angle))

                x_ = (c*(x-x0-x_drift*l3_seq)-s*(y-y0-y_drift*l3_seq))+x0

                rot_z = scale_factor[l3_cam]/octagon_size_pix*(x_ - octagon_edge_pix)+1

                vignette_correction[l3_seq,l3_cam] = rot_z

        return vignette_correction

    @staticmethod
    def default_pickle_path() -> pathlib.Path:
        return ov_final_path

    @classmethod
    def from_pickle(cls, path: typ.Optional[pathlib.Path] = None) -> 'Level3':
        obs = super().from_pickle(path)
        obs.observation.wcs.array_shape = obs.observation.data.shape
        return obs

    def to_fits(self, path: pathlib.Path):
        '''
        In need of a rework since moving to NDCube.  Note that WCS.to_header does not output naxis keywords
        '''

        path.mkdir(parents=True, exist_ok=True)

        for sequence in range(self.start_time.shape[0]):
            for camera in range(self.start_time.shape[1]):
                name = 'ESIS_Level3_' + str(self.start_time[sequence, camera]) + '_' + str(camera + 1) + '.fits'
                filename = path / name

                hdr = self.wcs[sequence].to_header()
                # hdr['CAM_ID'] = self.cam_id[sequence, camera]
                hdr['CAM_ID'] = camera+1  #place holder since I didn't add the real cam id to the current pickel
                hdr['DATE_OBS'] = str(self.start_time[sequence, camera])

                hdul = fits.HDUList()
                hdul.append(fits.PrimaryHDU(np.array(self.intensity[sequence, camera, ...]), hdr))
                hdul.writeto(filename, overwrite=True)

        output_file = path.name + '.tar.gz'
        with tarfile.open(path.parent / output_file, "w:gz") as tar:
            tar.add(path, arcname=path.name)

    def to_test_object(self, aia_path: pathlib.Path = default_aia_path, level1_path: pathlib.Path = level_1.Level1.default_pickle_path()) -> 'Level3':
        '''
        Replace all images in a Level 3 object with co-temporal AIA 304 images for testing.
        '''

        esis = level_1.Level1.from_pickle(level1_path)
        aia_304 = aia.AIA.from_path('aia_304', aia_path)
        transforms = img_align.TransformCube.from_pickle(self.transformation_objects)
        for l3_seq, l1_seq in enumerate(self.lev1_sequences):
            for l3_cam, l1_cam in enumerate(self.lev1_cameras):
                td = aia_304.exposure_start_time - esis.start_time[l1_seq, 0]  # should be the same for every camera
                best_im = np.abs(td.sec).argmin()
                aia_im = aia_304.intensity[best_im, ...]
                crop = transforms.transform_cube[l3_seq][l3_cam].post_transform_crop
                self.observation.data[l3_seq,l3_cam] = aia_im[crop]
        self.observation.meta = {'Description':'Test object using AIA 304 images and the same masks as initial Level3 object'}

        return self

    def masked_mean_normalization(self) -> np.ndarray:
        '''
        Given a masked level3 object this routine will return the mean taken from the portion of the sun seen by all 4
        cameras that does not contain the bright MgX line as a cube by which level3.observation.data can be divided.

        For best results correct vignetting first.
        '''

        super_mask = np.ones_like(self.observation.mask[:,0])
        for i in self.lev1_cameras:
            super_mask *= self.observation.mask[:, i]

        super_mask = np.resize(super_mask[:,None],self.observation.mask.shape)
        masked_cube = self.observation.data * super_mask
        masked_cube[masked_cube == 0] = np.nan
        mean_cube = np.nanmean(masked_cube,axis = (-2,-1),keepdims= True)

        return mean_cube











def vignetting_correction_quality(x,lev3: 'Level3'):

    scale_factor = x[0:4]
    vignetting = lev3.correct_vignetting(scale_factor)

    camera_combos = [[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
    # camera_combos = [[0,1],[0,2],[1,2]]

    sequences = [(i,j) for (i,j) in enumerate(lev3.lev1_sequences)]
    slope = []
    for lev3_seq,lev1_seq in sequences[::1]:

        mask1 = lev3.observation.mask[lev3_seq, 0]
        mask2 = lev3.observation.mask[lev3_seq, 1]
        mask3 = lev3.observation.mask[lev3_seq, 2]
        mask4 = lev3.observation.mask[lev3_seq, 3]
        masks = [mask1,mask2,mask3,mask4]
        super_mask = mask1*mask2*mask3*mask4

        im1 = lev3.observation.data[lev3_seq,0]  / vignetting[lev3_seq,0]
        im1 /= np.mean(im1[im1*super_mask != 0])
        im2 = lev3.observation.data[lev3_seq,1]  / vignetting[lev3_seq,1]
        im2 /= np.mean(im2[im2*super_mask != 0])
        im3 = lev3.observation.data[lev3_seq,2]  / vignetting[lev3_seq,2]
        im3 /= np.mean(im3[im3*super_mask != 0])
        im4 = lev3.observation.data[lev3_seq,3] / vignetting[lev3_seq,3]
        im4 /= np.mean(im4[im4*super_mask != 0])


        normalized_images = np.array([im1,im2,im3,im4])

        for in1,in2 in camera_combos:
            dif_im = normalized_images[in1]-normalized_images[in2]
            # fig,ax = plt.subplots()
            # ax.imshow(dif_im,vmin = -2,vmax = 2)
            # plt.show()

            # dif_im *= super_mask
            dif_im *= masks[in1]*masks[in2]
            dif_im[dif_im == 0] = np.nan

            column_mean = np.nanmean(dif_im,axis = 0)
            column_mean = column_mean[300:-300]
            # column_mean = column_mean[column_mean != 0]
            # fig,ax = plt.subplots()
            # ax.plot(column_mean)
            # plt.show()

            poly_fit = np.polynomial.Polynomial.fit(np.arange(column_mean.shape[0]),column_mean,deg = 1 )
            fit_slope = poly_fit.coef[1]
            slope.append(fit_slope)

    slope = np.array(slope)
    # slope_max = np.array(slope).max()
    # slope_total = np.sum(np.array(np.abs(slope)))
    least_squares = np.sqrt(np.sum(np.square(slope)))/slope.size
    print(least_squares)
    return least_squares

def find_vignetting_correction(lev3:'Level3'):
    # bounds = [(.2,.6),(.2,.6),(.2,.6),(.2,.6),(-10,10),(-10,10),(-10,10),(-10,10)]
    # bounds = [(.3, .5), (.3, .5), (.3, .5), (.3, .5), (-0, 0), (-0, 0), (-0, 0), (-0, 0)]
    # guess = np.array([0.38815087, 0.33319833, 0.32219223, 0.4400179 , 0.        ,
    #    0.        , 0.        , 0.        ])
    # bounds = [(.2, .6)]
    guess = np.array([.4,.4,.4,.5])
    # bounds = [(.4,.42),(-0,0),(-0,0),(-0,0),(-0,0)]

    bounds = [(.2,.8),(.2,.8),(.2,.8),(.2,.8)]
    fit = scipy.optimize.minimize(vignetting_correction_quality,guess,args = (lev3,),bounds = bounds, options={'ftol': 1e-5})
    # fit = scipy.optimize.differential_evolution(vignetting_correction_quality,bounds,args = (lev3,),polish = False)

    return fit


