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
from kgpy.moment.percentile import width
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


plt.rcParams.update({'font.size': 9})

def two_gauss(domain, a0, a1, sig0, sig1, shift0, shift1):
    gauss = a0 * np.exp(-np.square((domain - shift0) / sig0) / 2)
    gauss = gauss + a1 * np.exp(-np.square((domain - shift1) / sig1) / 2)
    return gauss


def one_gauss(domain, a0, sig0, shift0):
    gauss = a0 * np.exp(-np.square((domain - shift0) / sig0) / 2)
    return gauss


def gauss_fit(domain, data):
    # single gaussian fit
    og_peak = data.max()
    og_shift = domain[data.argmax()]
    og_width = width(data)[0] / 1.35  # 2.6 matches gaussian width more closely

    og_guess = [og_peak, og_width, og_shift]
    og_fit_params, fit_cov = scipy.optimize.curve_fit(one_gauss, domain, data, p0=og_guess)

    # double gaussian fit
    data_pad = np.pad(data, (1, 1))
    slope = data_pad - np.roll(data_pad, 1)
    dg_shift_1 = 0
    while slope[dg_shift_1] >= 0:
        dg_shift_1 += 1
    dg_peak_1 = data[dg_shift_1]
    dg_shift_1 = domain[dg_shift_1]
    dg_width_1 = og_width / 2

    dg_shift_2 = -1
    while slope[dg_shift_2] <= 0:
        dg_shift_2 -= 1
    dg_peak_2 = data[dg_shift_2]
    dg_shift_2 = domain[dg_shift_2]
    dg_width_2 = og_width / 2
    # dg_shift_1 = dg_shift_2
    if dg_shift_1 == dg_shift_2:
        qt = len(domain) // 4
        dg_shift_1 = domain[qt]
        dg_shift_2 = domain[-qt]
        dg_peak_1 = og_peak / 2
        dg_peak_2 = og_peak / 2

    dg_guess = [dg_peak_1, dg_peak_2, dg_width_1, dg_width_2, dg_shift_1, dg_shift_2]
    dg_fit_params, fit_cov = scipy.optimize.curve_fit(two_gauss, domain, data, p0=dg_guess)

    og_fit_error = np.sum(np.square(data - one_gauss(domain, *og_fit_params)))
    dg_fit_error = np.sum(np.square(data - two_gauss(domain, *dg_fit_params)))

    if og_fit_error < dg_fit_error:
        fit_params = [og_fit_params[0], 0, og_fit_params[1], 0, og_fit_params[2], 0]
    else:
        fit_params = dg_fit_params

    return fit_params





if __name__ == '__main__':
    event_pad = 7
    guass_fit_trim = 7
    dif_thresh = 50
    seq = 11
    int_max = 360
    lp_max = 30

    event = l3_events.perfectx
    l4 = level_4.Level_4.from_pickle(event.mart_inverted_pickle_path)
    l4_nofilter = level_4.Level_4.from_pickle(event.mart_inverted_pickle_path.parent / 'lev4_perfectx_mart_nofilter.pickle')
    l4_maxfilter = level_4.Level_4.from_pickle(event.mart_inverted_pickle_path.parent / 'lev4_perfectx_mart_maxfilter.pickle')

    l4s = [l4_nofilter,l4,l4_maxfilter]

    l4_int = l4.integrated_intensity
    brightest_pix = np.unravel_index(l4_int.argmax(), l4_int.shape)

    l3 = level_3.Level_3.from_pickle(level_3.ov_final_path)
    times = l3.time
    event_imgs = l3.observation.data[..., event.location[0], event.location[1]]
    dif = event_imgs[:, 1, ...] - event_imgs[:, 2, ...]

    event_pix = []
    spread = 2

    n = 0
    for i in [-spread, 0, spread]:
        n += 1
        for j in [-spread, 0, spread]:
            event_pix.append([brightest_pix[1] + i, brightest_pix[2] + j])

    event_pix = np.array(event_pix).T

    wr = [1, 1, 1, .05]
    fig1 = plt.figure(figsize=[7.1, 3])
    spec1 = gridspec.GridSpec(nrows=1, ncols= 4, width_ratios=wr)
    axs_top = []
    axs_bottom = []

    crop = (
        slice(brightest_pix[1] - event_pad, brightest_pix[1] + event_pad + 1),
        slice(brightest_pix[2] - event_pad, brightest_pix[2] + event_pad + 1))

    for j, lev4 in enumerate(l4s):
        int_wcs = lev4.wcs_list[seq].dropaxis(0).slice((crop[0], crop[1]))
        axs_top.append(fig1.add_subplot(spec1[0, j], projection=int_wcs))
        l4_int = lev4.integrated_intensity
        int_cutout = l4_int[seq, crop[0], crop[1]]
        intensity = axs_top[j].imshow(int_cutout, vmax=int_max)
        t = axs_top[j].annotate(times[seq].strftime('%H:%M:%S'), (.25, .25), color='w')
        t.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
        axs_top[j].coords[0].set_ticklabel(exclude_overlapping=True)
        axs_top[j].coords[0].set_axislabel('Solar X (arcsec)')
        axs_top[j].coords[1].set_axislabel('Solar Y (arcsec)')

        if j > 0:
            axs_top[j].coords[1].set_ticklabel_visible(False)
        if j == 2:
            cbaxes = fig1.add_subplot(spec1[0, j + 1])
            cb = plt.colorbar(intensity, cax=cbaxes)

        for i in range(event_pix.shape[1]):
            axs_top[j].plot(event_pix[1, i] - crop[1].start, event_pix[0, i] - crop[0].start, marker='.', color='r')

    axs_top[0].set_title('Integrated Intensity \n Filter Steps = 0')
    axs_top[1].set_title('Integrated Intensity \n Filter Steps = 25')
    axs_top[2].set_title('Integrated Intensity \n Filter Steps = 75')


    fig2, axs = plt.subplots(n, n, constrained_layout=True, figsize=(7.1, 5))
    axs = axs[::-1]  # corrects subplots to match dot locations
    flat_axs = axs.flatten()
    colors = ['r', 'g', 'b']

    fits = []

    for j, lev4 in enumerate(l4s):
        for i in range(event_pix.shape[1]):
            line_profile = np.array(lev4.cube_list)[seq, event_pix[0, i], event_pix[1, i], :]
            domain = np.array(lev4.velocity_axis)

            p0 = [line_profile.max(), line_profile.max() / 2, 50, 50, 100, -100]
            # p0 = [14, 4, 50, 50, -90, 110]

            bounds = (
                [0, 0, 15, 15, -200, -200], [line_profile.max() * 1.2, line_profile.max() * 1.2, 100, 100, 200, 200])
            fit_params, fit_cov = scipy.optimize.curve_fit(two_gauss, domain[guass_fit_trim: -guass_fit_trim],
                                                           line_profile[guass_fit_trim: -guass_fit_trim], p0=p0,
                                                           bounds=bounds)
            # fit_params = gauss_fit(domain[guass_fit_trim: -guass_fit_trim], line_profile[guass_fit_trim: -guass_fit_trim])
            fits.append(fit_params)

            flat_axs[i].plot(domain, line_profile, marker='.', linestyle='None', color=colors[j], ms=4)
            flat_axs[i].plot(domain, two_gauss(domain, *fit_params), color=colors[j], linewidth=1)
            flat_axs[i].set_ylim(top=lp_max)
            flat_axs[i].xaxis.set_minor_locator(AutoMinorLocator())
            w1 = '%.0f' % fit_params[2]
            w2 = '%.0f' % fit_params[3]
            vel1 = '%.0f' % fit_params[4]
            vel2 = '%.0f' % fit_params[5]
            flat_axs[i].annotate('v =' + vel1 + ', ' + vel2, (130, lp_max - 2.5*(j+1)), color=colors[j],
                                 fontsize=7)
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

    filepath1 = event.name + '_invert_comp_a.pdf'
    filepath2 = event.name + '_invert_comp_b.pdf'
    fig1.savefig(fig_path / filepath1)
    fig2.savefig(fig_path / filepath2)
