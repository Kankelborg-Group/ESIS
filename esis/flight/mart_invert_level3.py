import numpy as np
from esis.data import level_3, level_4
import matplotlib.pyplot as plt
import scipy.ndimage
from esis.data.inversion import mart
import astropy.units as u
import kgpy.img.coalignment.image_coalignment as kgpy_img
import astropy.wcs as wcs
import multiprocessing as mp
import skimage.transform
from kgpy.multiprocess_tools.mp_wrapper import starmap_with_kwargs
from itertools import repeat
import time

if __name__ == '__main__':

    # plt.rcParams['figure.figsize'] = [20, 20]

    ov = level_3.Level3.from_pickle(level_3.ov_final_path)
    ov_data = ov.observation.data
    test_seq = 15

    # rotation_kwargs = {
    #     'reshape': False,
    #     'prefilter': False,
    #     'order': 1,
    #     'mode': 'constant',
    #     'cval': 0,
    # }

    angles = (np.arange(4)*45 - 22.5 + 45 - 90)* u.deg
    # event = [slice(None),slice(None)]
    #
    # # event = [slice(80,1120),slice(80,1120)]
    # pad = 0
    # # #
    # #
    event = [slice(400,700),slice(550,850)]
    pad = 20

    region = ov_data[:,:,event[0],event[1]]


    x,y = kgpy_img.get_img_coords(region[0,0])
    x0,y0 = [region.shape[-2] // 2, region.shape[-1] //2]
    len_scl = min(region.shape[-1],region.shape[-2])/2.5
    window = np.exp(-(np.sqrt(np.square(x-x0) + np.square(y-y0))/len_scl)**(6))
    window=window.T
    # window = np.ones_like(region[0])

    guess = np.ones_like(region[0,0])*window
    guess = np.pad(guess,((pad,pad),(pad,pad)))

    guess = guess[None,:,:]

    guess = np.resize(guess,(41,guess.shape[-2],guess.shape[-1]))
    guess = np.moveaxis(guess,0,-1)


    spectral_order = 1
    mart_obj = mart.MART(
        use_maximize=True,
        use_filter=True,
        use_lgof = True,
        anti_aliasing=None,
        max_multiplicative_iteration=100,
        max_filtering_iterations=75,
        photon_read_noise = 2,
        # track_cube_history='filter',
        contrast_exponent=.4,
        # rotation_kwargs=rotation_kwargs

    )

    ref_wavelen = guess.shape[-1] // 2
    recovered_list = []

    # for seq in range(ov.observation.data.shape[0]-3):
    seqs = [i for i in range(ov.observation.data.shape[0])]
    seqs = [0,1,2,3,4,5,6,7]

    projections_list = []
    for seq in seqs:
        projections = []
        for i,angle in enumerate(angles):
            # projection = scipy.ndimage.rotate(np.pad(region[seq,i]*window,((pad,pad),(pad,pad))),angle,**rotation_kwargs)
            projection = skimage.transform.rotate(np.pad(region[seq,i]*window,((pad,pad),(pad,pad))),angle.value)
            projections.append(projection)


        projections = np.array(projections)
        projections = projections[None,:,:,:,None]
        projections[projections<0] = 0
        projections_list.append(projections)




    p = mp.Pool(mp.cpu_count()//2)
    # p = mp.Pool(len(seqs))
    args_iter = zip(projections_list,repeat(angles),repeat(np.array(spectral_order)))
    kwargs_iter = repeat(dict(cube_offset_x=ref_wavelen,cube_guess=guess))

    start = time.time()
    recovered_list = starmap_with_kwargs(p,mart_obj,args_iter,kwargs_iter)
    print('Total Inversion Time = ',time.time()-start)

    image_wcs = ov.observation[0,0,event[0],event[1]].wcs.dropaxis(-1)
    image_wcs = image_wcs.dropaxis(-1)
    print(ref_wavelen)

    header = image_wcs.to_header()
    header['ctype1'] = 'Solar Y'
    header['ctype2'] = 'Solar X'
    header['naxis'] = 3
    header['ctype3'] = 'pix'
    header['crval3'] = -18
    header['crpix3'] = ref_wavelen
    header['cdelt3'] = 18
    header['naxis3'] = guess.shape[-1]

    result_wcs = wcs.WCS(header)
    result_wcs = result_wcs.swapaxes(-1,0)
    result_wcs.array_shape = guess.shape


    inverted_results = np.array([recovered_list[i].best_cube for i in range(len(recovered_list))])
    inverted_results_wcs = [result_wcs for i in range(len(recovered_list))]


    lev4 = level_4.Level_4(inverted_results,inverted_results_wcs)
    lev4.to_pickle(path = 'lev4_mainevent_mart.pickle')

    test = lev4.plot()
    plt.show()


