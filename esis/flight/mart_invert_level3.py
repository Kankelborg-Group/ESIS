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
from esis.flight import l3_events
from kgpy.plot import CubeSlicer

if __name__ == '__main__':

    # plt.rcParams['figure.figsize'] = [20, 20]
    start = time.time()
    ov = level_3.Level_3.from_pickle(level_3.ov_final_path)
    ov_data = ov.observation.data
    test_seq = 15

    # slice = CubeSlicer(ov_data[:,0,...])
    # plt.show()
    # rotation_kwargs = {
    #     'reshape': False,
    #     'prefilter': False,
    #     'order': 1,
    #     'mode': 'constant',
    #     'cval': 0,
    # }

    angles = (np.arange(4) * 45 - 22.5 + -45) * u.deg
    print(angles)

    l3_event = l3_events.perfectx

    save_path = 'lev4_' + l3_event.name + '_mart.pickle'
    event = l3_event.location
    pad = 50
    event = (slice(event[0].start - pad, event[0].stop + pad, None), slice(event[1].start - pad, event[1].stop + pad))


    extent_y = event[0].stop - event[0].start
    extent_x = event[1].stop - event[1].start

    pad = np.max(np.sqrt(extent_x ** 2 + extent_y ** 2) - (extent_x, extent_y))
    pad = int(np.ceil(pad * 1.05))


    region = ov_data[:, :, event[0], event[1]]

    x, y = kgpy_img.get_img_coords(region[0, 0])
    x0, y0 = [region.shape[-2] // 2, region.shape[-1] // 2]
    len_scl = min(region.shape[-1], region.shape[-2]) / 2.5
    window = np.exp(-(np.sqrt(np.square(x - x0) + np.square(y - y0)) / len_scl) ** (6))
    window = window.T
    window = np.ones_like(window)

    guess = np.ones_like(region[0, 0]) * window
    guess = np.pad(guess, ((pad, pad), (pad, pad)))

    guess = guess[None, :, :]

    guess = np.resize(guess, (41, guess.shape[-2], guess.shape[-1]))
    guess = np.moveaxis(guess, 0, -1)


    spectral_order = 1
    mart_obj = mart.MART(
        use_maximize=False,
        use_filter=True,
        use_lgof=False,
        max_multiplicative_iteration=20,
        max_filtering_iterations=25,
        photon_read_noise=2,
        # track_cube_history='filter',
        contrast_exponent=.2,
        # rotation_kwargs=rotation_kwargs
        verbose=True

    )

    ref_wavelen = guess.shape[-1] // 2
    recovered_list = []


    seqs = [i for i in range(ov.observation.data.shape[0])]
    # seqs = [13, 14, 15, 16]
    seqs = [15]

    channels = [0, 1, 2, 3]
    # channels = [1,2]



    projections_list = []
    for seq in seqs:
        projections = []
        for chan in channels:
            # projection = scipy.ndimage.rotate(np.pad(region[seq,i]*window,((pad,pad),(pad,pad))),angle,**rotation_kwargs)
            projection = skimage.transform.rotate(np.pad(region[seq, chan] * window, ((pad, pad), (pad, pad))),
                                                  angles[chan].value)
            projections.append(projection)
            # fig, ax = plt.subplots()
            # ax.imshow(projection)
            # plt.show()

        projections = np.array(projections)
        projections = projections[None, :, :, :, None]
        projections[projections < 0] = 0
        projections_list.append(projections)


    used_angles = [angles[i] for i in channels] * u.deg
    p = mp.Pool(mp.cpu_count() - 2)
    # p = mp.Pool(len(seqs))
    args_iter = zip(projections_list, repeat(used_angles), repeat(np.array(spectral_order)))
    kwargs_iter = repeat(dict(cube_offset_x=ref_wavelen, cube_guess=guess))

    start = time.time()
    recovered_list = starmap_with_kwargs(p, mart_obj, args_iter, kwargs_iter)
    print('Total Inversion Time = ', time.time() - start)

    image_wcs = ov.observation[0, 0, event[0], event[1]].wcs.dropaxis(-1)
    image_wcs = image_wcs.dropaxis(-1)
    print(ref_wavelen)

    # plt.figure(dpi=140)
    # plt.plot(recovered_list[0].norm_history)
    # plt.show()

    inverted_results = np.array(
        [recovered_list[i].best_cube[pad:-pad, pad:-pad, ...] for i in range(len(recovered_list))])

    velocity_correction = (.6 / .74)    #comes from rebining to AIA resolution of .6 arcsec per pix
    header = image_wcs.to_header()
    header['naxis'] = 3
    header['ctype3'] = 'km / s'
    header['crval3'] = -18 * velocity_correction
    header['crpix3'] = ref_wavelen
    header['cdelt3'] = 18 * velocity_correction
    header['naxis3'] = inverted_results[0].shape[-1]

    result_wcs = wcs.WCS(header)
    result_wcs = result_wcs.swapaxes(-1,-2).swapaxes(-2,-3)
    result_wcs.array_shape = inverted_results[0].shape
    print(result_wcs)

    inverted_results_wcs = [result_wcs for i in range(len(recovered_list))]

    lev4 = level_4.Level_4(inverted_results, inverted_results_wcs)
    print('Inversion Duration = ', start=time.time())
    # lev4.to_pickle(path =save_path)

    test = lev4.plot()
    plt.show()
