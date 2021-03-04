import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import kgpy.img.coalignment.image_coalignment as img_align
from kgpy.observatories.sdo import aia
from esis.data import level_3, level_4
import astropy.units as u
import scipy.ndimage
import pathlib
from esis.flight import l3_events

if __name__ == '__main__':

    event = l3_events.main_event

    lev3 = level_3.Level_3.from_pickle(level_3.ov_final_path)
    times = lev3.time

    l4 = level_4.Level_4.from_pickle(event.mart_inverted_pickle_path)
    l4_int = l4.integrated_intensity

    peak_profile = np.array(l4.cube_list).max()

    shifts = l4.shifts(95)
    widths = l4.widths(95)

    event_location = event.location

    l3_event_imgs = lev3.observation.data[:, 1, event_location[0], event_location[1]] - lev3.observation.data[:, 2, event_location[0], event_location[1]]

    ### NOTE likely a bug in Level 3 wcs that requires a swap axes here
    l3_img_wcs = lev3.observation[:, 0, event_location[0], event_location[1]].wcs.dropaxis(-1).dropaxis(-1).swapaxes(1, 0)

    aia_304 = aia.AIA.from_time_range(times[0], times[-1] + 100 * u.s, channels=[304 * u.AA], user_email='jacobdparker@gmail.com')

    aia_304_imgs = []
    l3_imgs = []
    frames = []
    for seq,l1_seq in enumerate(lev3.lev1_sequences):
        l3_img = np.squeeze(l3_event_imgs[seq])
        l3_imgs.append(l3_img)

        coords = img_align.get_img_coords(l3_img)
        coord_shp = coords.shape
        coords = coords.reshape(coord_shp[0], coord_shp[1] * coord_shp[2])
        l3_world_coords = l3_img_wcs.all_pix2world(coords.T, 0)

        aia_wcs = aia_304.wcs[seq, 0]
        aia_pix = aia_wcs.all_world2pix(l3_world_coords, 0).T
        aia_pix = aia_pix.reshape(coord_shp)

        aia_304_img = scipy.ndimage.map_coordinates(aia_304.intensity[seq, 0], aia_pix).T
        aia_304_imgs.append(aia_304_img)

        frames.append(seq)


    #NOTE:  Worst plotting ever.  If there is a solution that makes all 6 of these the same size PLEASE tell me.
    fig = plt.figure(figsize = (7,14))
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[0], projection=l3_img_wcs, title = 'Level 3 Difference (2 - 3)')
    ax1.set_xlabel('Solar X (arcsec)')
    ax1.set_ylabel('Solar Y (arcsec)')

    ax2 = fig.add_subplot(gs[1], projection=l3_img_wcs, title = 'Inverted LOS Doppler Velocity')
    ax2.set_xlabel('Solar X (arcsec)')
    ax2.set_ylabel('Solar Y (arcsec)')

    ax3 = fig.add_subplot(gs[2], projection=l3_img_wcs, title = 'Summed Intensity (Inverted Results)')
    ax3.set_xlabel('Solar X (arcsec)')
    ax3.set_ylabel('Solar Y (arcsec)')

    ax4 = fig.add_subplot(gs[4], projection=l3_img_wcs, title = 'AIA 304 Angstrom')
    ax4.set_xlabel('Solar X (arcsec)')
    ax4.set_ylabel('Solar Y (arcsec)')

    ax5 = fig.add_subplot(gs[3], projection=l3_img_wcs, title = 'Inverted Line Width')
    ax5.set_xlabel('Solar X (arcsec)')
    ax5.set_ylabel('Solar Y (arcsec)')

    ax6 = fig.add_subplot(gs[5], projection = l4.wcs_list[0].dropaxis(-1).dropaxis(-1), title='Line Profiles (by dot color)')
    ax6.set_ylabel('Intensity')
    ax6.set_ylim([0,60])

    p = 99
    im1 = ax1.imshow(l3_imgs[0])
    t = ax1.annotate(times[0],(2,2), color='w')

    pix1 = (25,54)
    pix2 = (52,55)
    pix3 = (48,40)

    im2 = ax2.imshow(shifts[0]*18,cmap = 'RdBu_r',vmin = -100, vmax = 100)
    im3 = ax3.imshow(l4_int[0], vmax=np.percentile(np.array(l4_int), p), origin='lower')
    point1 = ax3.plot(pix1[1],pix1[0], marker='.')
    point2 = ax3.plot(pix2[1],pix2[0], marker='.')
    point3 = ax3.plot(pix3[1],pix3[0], marker='.')

    im4 = ax4.imshow(aia_304_imgs[0], vmax=np.percentile(np.array(aia_304_imgs[0]), p), cmap='sdoaia304')
    im5 = ax5.imshow(widths[0]*18, vmax = 150)

    line1 = ax6.plot(l4.cube_list[0][pix1])
    line2 = ax6.plot(l4.cube_list[0][pix2])
    line3 = ax6.plot(l4.cube_list[0][pix3])

    fig.colorbar(im2, ax=ax2, label='km/s', fraction=0.046, pad=0.04)
    fig.colorbar(im5, ax=ax5, label='km/s', fraction=0.046, pad=0.04)

    plt.subplots_adjust(hspace = .4)


    def update(seq):
        im1.set_data(l3_imgs[seq])
        t.set_text(times[seq])
        im2.set_data(shifts[seq]*18)
        im3.set_data(l4_int[seq])
        im4.set_data(aia_304_imgs[seq])
        im5.set_data(widths[seq]*18)
        line1[0].set_ydata(l4.cube_list[seq][pix1])
        line2[0].set_ydata(l4.cube_list[seq][pix2])
        line3[0].set_ydata(l4.cube_list[seq][pix3])

    ani = FuncAnimation(fig, update, frames = frames)
    plt.show()

    save_path = pathlib.Path(__file__).parent / 'figures/' / 'main_event_movie.avi'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(save_path,'ffmpeg',dpi = 200)