from esis.data import level_3, level_4
import numpy as np
import matplotlib.pyplot as plt
from esis.flight import l3_events
import skimage.transform
import astropy.units as u
import kgpy.img.coalignment.image_coalignment as img_align
import esis.flight
from esis.science.papers.mission_paper import fig_path  # can't figure out relative import here

if __name__ == '__main__':
    #takes a quick slice through an event in L1 starting from a bright pixel in L4.  Maps from inversion, to l3 to l1

    event = l3_events.perfectx
    l4 = level_4.Level_4.from_pickle(event.mart_inverted_pickle_path)
    l3 = level_3.Level_3.from_pickle(level_3.ov_final_path)
    l1 = esis.flight.level_1()
    l1.intensity = np.flip(l1.intensity, axis=-2)

    l4_int = l4.integrated_intensity




    seq = 11
    brightest_pix = np.unravel_index(l4_int[seq].argmax(), l4_int[seq].shape)
    cams = [0, 1, 2, 3]

    # loc = np.array([(brightest_pix[1],), (brightest_pix[2],)]).T
    loc = np.array([(brightest_pix[0],), (brightest_pix[1],)]).T
    l4_world_coords = l4.wcs_list[seq].dropaxis(0).all_pix2world(loc, 0)
    l3_pix = l3.observation.wcs.dropaxis(-1).dropaxis(-1).all_world2pix(l4_world_coords, 0)
    l3_pix = l3_pix.reshape((2, 1))
    lev1_transforms = img_align.TransformCube.from_pickle(l3.transformation_objects).transform_cube
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots(2, 2, constrained_layout=True)
    axs = ax2.flatten()
    for i, cam in enumerate(cams):
        transform = lev1_transforms[seq][cam]
        l1_img = l1.intensity[l3.lev1_sequences[seq], cam, ...].value

        coords = transform.coord_post_process(l3_pix[..., None], reverse=True)

        dummy_img = np.empty((1, 1))
        coords = transform.transform_func(dummy_img, transform.transform, transform.origin, old_coord=coords[::-1])
        coords = coords[::-1]

        coords = transform.coord_pre_process(coords, reverse=True)
        coords = coords.squeeze()
        round_coords = coords.round().astype(int)
        print(coords,round_coords)

        # fig, ax = plt.subplots()
        # ax.imshow(l1_img, vmax = np.percentile(l1_img,99.9), origin='lower')

        lp_sl = slice(round_coords[0] - 21, round_coords[0] + 21)
        v_sl = slice(round_coords[1] - 4, round_coords[1] + 3)
        print(v_sl)
        l1_slice = np.arange(lp_sl.start, lp_sl.stop)
        domain = (l1_slice - coords[0]) * 18

        lps = l1_img[v_sl, lp_sl]
        print(lps.shape)
        # lp = lps.sum(axis=0)/lps.shape[0]
        lp = l1_img[round_coords[1], lp_sl]
        lp_plot = ax1.plot(domain, lp)
        ax1.set_xlabel('Velocity (km/s)')
        ax1.set_ylabel('Counts (DN)')

        l1_cutout = (slice(v_sl.start - 15, v_sl.stop + 15), slice(lp_sl.start - 5, lp_sl.stop + 5))
        cropped_l1 = l1_img[l1_cutout]
        axs[i].imshow(cropped_l1, origin='lower', vmax=np.percentile(cropped_l1, 99.5))
        x = round_coords[0] - l1_cutout[1].start
        y = round_coords[1] - l1_cutout[0].start


        axs[i].plot(x, y, color=lp_plot[0].get_color(), marker='.', ms=5)
        axs[i].plot(l1_slice - l1_cutout[1].start, [y] * len(l1_slice), color=lp_plot[0].get_color())
        axs[i].set_title('Camera '+str(cam+1) + ' Level-1')

    axs[3].set_xlabel('Level-1 Pixels')
    axs[2].set_xlabel('Level-1 Pixels')
    axs[0].set_ylabel('Level-1 Pixels')
    axs[2].set_ylabel('Level-1 Pixels')
    plt.show()

    filepath1 = event.name + '_slice_lps.pdf'
    filepath2 = event.name + '_slice_imgs.pdf'
    fig1.savefig(fig_path / filepath1)
    fig2.savefig(fig_path / filepath2)



    # region = l3.observation.data[:, :, event.location[0], event.location[1]]
    # channels = [0,1,2,3]
    # angles = (np.arange(4) * 45 - 22.5 + -45) * u.deg
    # seq = 15
    # projections = []
    # for chan in channels:
    #     # projection = scipy.ndimage.rotate(np.pad(region[seq,i]*window,((pad,pad),(pad,pad))),angle,**rotation_kwargs)
    #     projection = skimage.transform.rotate(region[seq, chan],
    #                                           angles[chan].value)
    #     projections.append(projection)
    #     fig, ax = plt.subplots()
    #     ax.imshow(projection, origin='lower')
    #     plt.show()
    #
    # projections = np.array(projections)
    # projections = projections[None, :, :, :, None]
    # projections[projections < 0] = 0
