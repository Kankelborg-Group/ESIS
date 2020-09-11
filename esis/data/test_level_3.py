import numpy as np
import pathlib
import astropy.io.fits
import matplotlib.pyplot as plt

from kgpy.img.coalignment import image_coalignment
from esis.data import level_1, level_3


def test_to_fits():
    '''
    Likely needs serious rework since migration to NDCube for Level3
    '''
    path = level_3.ov_Level3_initial
    lev3 = level_3.Level3.from_pickle(path)

    dir = pathlib.Path(__file__).parent / 'ESIS_Level3_Fits/'

    lev3.to_fits(dir)

    p = list(dir.glob('*.fits'))
    p = p[0]

    f = astropy.io.fits.open(p)
    f.verify('fix')

def test_from_pickle():
    path = level_3.ov_Level3_initial
    lev3 = level_3.Level3.from_pickle(path)

    print(lev3.start_time)

def test_to_pickle(capsys):
    with capsys.disabled():
        path = level_3.ov_Level3_initial
        lev3 = level_3.Level3.from_pickle(path)
        lev3.to_pickle(level_3.ov_Level3_initial)

def test_from_aia_level1(capsys):
    with capsys.disabled():
        lev3 = level_3.Level3.from_aia_level1()
        lev3.to_pickle(level_3.ov_Level3_initial)

def test_update_internal_alignment(capsys):
    with capsys.disabled():
        lev3 = level_3.Level3.from_pickle(level_3.ov_Level3_initial)
        lev3_update = lev3.update_internal_alignment()
        lev3_update.to_pickle(level_3.ov_Level3_updated)


def test_add_mask():
    lev3 = level_3.Level3.from_pickle(level_3.ov_Level3_updated)
    lev3.add_mask(line= 'mgx')

    cube_slice = lev3.observation[12,3,:,:]
    masked_img = cube_slice.data * cube_slice.mask

    fig,ax = plt.subplots()
    ax.imshow(masked_img)

    lev3.to_pickle(level_3.ov_Level3_masked)

def test_vignetting_correction(capsys):
    with capsys.disabled():
        lev3 = level_3.Level3.from_pickle(level_3.ov_Level3_masked)

        fit = level_3.find_vignetting_correction(lev3)
        print(fit)


        # fit = np.array([0.40371635, 4.46974554, -0.6823358, 7.80875192, -4.22512385])
        # fit = np.array([ 0.40843749,  0.35517264,  0.51301844,  0.4767043 , -8.81371075,
        #        -1.15916574,  8.57623802,  8.99914776])
        # vignetting = lev3.correct_vignetting(fit[0],fit[1:])

    # fit = level_3.vignetting_correction_quality(np.array([.40167,5.9317,.0701,8.2418,-5.80780]),lev3)
    # fit = level_3.vignetting_correction_quality(np.array([ 0.40371635,  4.46974554, -0.6823358 ,  7.80875192, -4.22512385]),lev3)

    # fit = level_3.vignetting_correction_quality(np.array([.40167,0,0,0,0]),lev3)


def test_transform_image():

    lev1 = level_1.Level_1.from_pickle()
    lev3 = level_3.Level3.from_pickle(level_3.ov_Level3_masked)
    transform_path = lev3.transformation_objects

    transformations = image_coalignment.TransformCube.from_pickle(transform_path)


    lev3_camera = 1
    lev3_sequence = 15

    fig, ax = plt.subplots()
    lev3_img = lev3.observation.data[lev3_sequence, lev3_camera, ...]
    ax.imshow(lev3_img)

    lev1_img_transform = transformations.transform_cube[lev3_sequence][lev3_camera]
    print(lev1_img_transform)

    lev1_sequence = lev3.lev1_sequences[lev3_sequence]
    lev1_camera = lev3.lev1_cameras[lev3_camera]

    lev1_img = lev1.intensity[lev1_sequence, lev1_camera, ...]


    lev1_img= lev1_img_transform.transform_image(lev1_img)


    fig, ax = plt.subplots()
    ax.imshow(lev1_img)

    fig, ax = plt.subplots()
    ax.imshow(lev1_img-lev3_img)
    ax.set_title('Should Be Zeros')

    plt.show()

