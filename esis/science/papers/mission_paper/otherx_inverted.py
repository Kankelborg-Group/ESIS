import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from esis.data import level_3, level_4
import pathlib
from esis.flight import l3_events
import scipy.optimize
import scipy.signal
from kgpy.plot import CubeSlicer
from kgpy.img import event_selector
import matplotlib.gridspec as gridspec
import astropy.units as u
import matplotlib.patches as patches
from esis.science.papers.mission_paper import fig_path  # can't figure out relative import here
import matplotlib.patheffects as PathEffects

plt.rcParams.update({'font.size': 9})


def two_gauss(domain, a0, a1, sig0, sig1, shift0, shift1):
    gauss = a0 * np.exp(-np.square((domain - shift0) / sig0) / 2)
    gauss = gauss + a1 * np.exp(-np.square((domain - shift1) / sig1) / 2)
    return gauss


if __name__ == '__main__':

    event = l3_events.otherx
    seqs = [6, 10, 15, 19]
    # seqs = [7, 8, 9, 10]

    l4 = level_4.Level_4.from_pickle(event.mart_inverted_pickle_path)
    # plot = l4.plot()
    # plt.show()
    l4_int = l4.integrated_intensity
    brightest_pix = np.unravel_index(l4_int.argmax(), l4_int.shape)
    print(brightest_pix)

    l3 = level_3.Level_3.from_pickle(level_3.ov_final_path)
    times = l3.time
    event_imgs = l3.observation.data[..., event.location[0], event.location[1]]
    dif = event_imgs[:,1,...] - event_imgs[:, 2, ...]

    event_pix = []
    spread = 2
    n = 0

    for i in [-spread, 0, spread]:
        n += 1
        for j in [-spread, 0, spread]:
            event_pix.append([brightest_pix[1] + i, brightest_pix[2] + j])

    event_pix = np.array(event_pix).T

    wr = [1, 1, 1, 1, .05]
    hr = [1, 1]
    fig1 = plt.figure(figsize=[7, 5])
    spec1 = gridspec.GridSpec(nrows=2, ncols=len(seqs) + 1, width_ratios=wr, height_ratios=hr)
    axs_top = []
    axs_bottom = []

    trim = 7
    crop = (
        slice(brightest_pix[1] - trim, brightest_pix[1] + trim + 1),
        slice(brightest_pix[2] - trim, brightest_pix[2] + trim + 1))
    for j, seq in enumerate(seqs):
        int_wcs = l4.wcs_list[seq].dropaxis(0).slice((crop[0], crop[1]))
        axs_top.append(fig1.add_subplot(spec1[0, j], projection=int_wcs))
        int_cutout = l4_int[seq, crop[0], crop[1]]
        intensity = axs_top[j].imshow(int_cutout, vmax=250)
        t = axs_top[j].annotate(times[seq].strftime('%H:%M:%S'), (.25, .25), color='w')
        t.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
        axs_top[j].coords[0].set_ticklabel(exclude_overlapping=True)
        axs_top[j].coords[0].set_axislabel('Solar X (arcsec)')
        axs_top[j].coords[1].set_axislabel('Solar Y (arcsec)')

        if j > 0:
            axs_top[j].coords[1].set_ticklabel_visible(False)
        if j == len(seqs) - 1:
            cbaxes = fig1.add_subplot(spec1[0, j + 1])
            cb = plt.colorbar(intensity, cax=cbaxes)

        for i in range(event_pix.shape[1]):
            axs_top[j].plot(event_pix[1, i] - crop[1].start, event_pix[0, i] - crop[0].start, marker='.', color='r')

        dif_wcs = l3.observation[0, j, event.location[0], event.location[1]].wcs.dropaxis(-1).dropaxis(-1)
        axs_bottom.append(fig1.add_subplot(spec1[1, j],
                                           projection=dif_wcs))
        dif_bound = 50
        dif_im = axs_bottom[j].imshow(dif[seq], origin='lower', vmin=-dif_bound, vmax=dif_bound)
        # dif_im = axs_bottom[j].imshow(dif[seq], origin='lower')
        axs_bottom[j].coords[0].set_ticklabel(exclude_overlapping=True)
        axs_bottom[j].coords[0].set_axislabel('Solar X (arcsec)')
        axs_bottom[j].coords[1].set_axislabel('Solar Y (arcsec)')
        if j > 0:
            axs_bottom[j].coords[1].set_ticklabel_visible(False)
        if j == len(seqs) - 1:
            cbaxes = fig1.add_subplot(spec1[1, j + 1])
            cb = plt.colorbar(dif_im, cax=cbaxes)

        shp = int_cutout.shape
        points = np.array([(0, 0), (0, shp[1]), (shp[0], shp[1]), (shp[0], 0)])
        l3_world_coords = int_wcs.all_pix2world(points, 0)
        dif_pix = dif_wcs.all_world2pix(l3_world_coords, 0)

        lw = 2
        box = patches.Polygon(dif_pix,
                              closed=True, edgecolor='r', facecolor='none', linewidth=lw,
                              )
        axs_bottom[j].add_patch(box)

    fits = []
    fig2, axs = plt.subplots(n, n, constrained_layout=True, figsize=(7.5, 6))
    axs = axs[::-1]  # corrects subplots to match dot locations
    flat_axs = axs.flatten()
    colors = ['r', 'g', 'b', 'black']

    for j, seq in enumerate(seqs):
        for i in range(event_pix.shape[1]):
            line_profile = np.array(l4.cube_list)[seq, event_pix[0, i], event_pix[1, i], :]

            p0 = [line_profile.max(), line_profile.max() / 2, 50, 50, 100, -100]
            # p0 = [14, 4, 50, 50, -90, 110]

            bounds = (
                [0, 0, 15, 15, -200, -200], [line_profile.max() * 1.2, line_profile.max() * 1.2, 100, 100, 200, 200])
            domain = np.array(l4.velocity_axis)

            trim = 7
            fit_params, fit_cov = scipy.optimize.curve_fit(two_gauss, domain[trim: -trim],
                                                           line_profile[trim: -trim], p0=p0, bounds=bounds)
            fits.append(fit_params)

            flat_axs[i].plot(domain, line_profile, marker='.', linestyle='None', color=colors[j])
            flat_axs[i].plot(domain, two_gauss(domain, *fit_params), color=colors[j])
            top = 19
            flat_axs[i].set_ylim(top=top)
            w1 = '%.0f' % fit_params[2]
            w2 = '%.0f' % fit_params[3]
            vel1 = '%.0f' % fit_params[4]
            vel2 = '%.0f' % fit_params[5]
            flat_axs[i].annotate('v =' + vel1 + ', ' + vel2, (130, top - 1.5 - 1.75 * j), color=colors[j], fontsize=7)
            # flat_axs[i].annotate('w =' + w1 + ',' + w2, (150, top - 2 - 1 * j), color=colors[j], fontsize=7)

            if i == 1 or i == 2 or i == 4 or i == 5 or i == 7 or i == 8:
                flat_axs[i].tick_params(labelleft=False)
            if i == 6 or i == 7 or i == 8 or i == 3 or i == 4 or i == 5:
                flat_axs[i].tick_params(labelbottom=False)

    flat_axs[0].set_xlabel('Velocity (km/s)')
    flat_axs[1].set_xlabel('Velocity (km/s)')
    flat_axs[2].set_xlabel('Velocity (km/s)')

    axs[0, 0].set_ylabel('Intensity (photons)')
    axs[1, 0].set_ylabel('Intensity (photons)')
    axs[2, 0].set_ylabel('Intensity (photons)')

    plt.show()
    fig1.savefig(fig_path / 'other_x_inverta.pdf')
    fig2.savefig(fig_path / 'other_x_invertb.pdf')
    plt.show()
