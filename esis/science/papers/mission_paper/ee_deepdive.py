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


def ee_deepdive_figures(event, seqs, event_pad, guass_fit_trim, dif_thresh):
    l4 = level_4.Level_4.from_pickle(event.mart_inverted_pickle_path)
    # plot = l4.plot()
    # plt.show()
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

    wr = [1, 1, 1, 1, .05]
    hr = [1, 1]

    # fig1 = plt.figure(figsize=[7.1, 3.3])
    fig1 = plt.figure(figsize=[7.1, 3.5])
    spec1 = gridspec.GridSpec(nrows=2, ncols=len(seqs) + 1, width_ratios=wr, height_ratios=hr, top=.99)
    axs_top = []
    axs_bottom = []

    crop = (
        slice(brightest_pix[1] - event_pad, brightest_pix[1] + event_pad + 1),
        slice(brightest_pix[2] - event_pad, brightest_pix[2] + event_pad + 1))

    int_max = []
    for seq in seqs:
        int_max.append(l4_int[seq, crop[0], crop[1]].max())
    int_max = np.array(int_max).max()

    for j, seq in enumerate(seqs):
        int_wcs = l4.wcs_list[seq].dropaxis(0).slice((crop[0], crop[1]))
        axs_top.append(fig1.add_subplot(spec1[0, j], projection=int_wcs))
        int_cutout = l4_int[seq, crop[0], crop[1]]
        intensity = axs_top[j].imshow(int_cutout, vmax=int_max)
        t = axs_top[j].annotate(times[seq].strftime('%H:%M:%S'), (.25, .25), color='w')
        t.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
        axs_top[j].coords[0].set_ticklabel(exclude_overlapping=True)
        axs_top[j].coords[0].set_axislabel(' ')
        # axs_top[j].set_xlabel('')
        axs_top[j].coords[1].set_axislabel('Solar Y (arcsec)')

        if j > 0:
            axs_top[j].coords[1].set_ticklabel_visible(False)
        if j == len(seqs) - 1:
            cbaxes = fig1.add_subplot(spec1[0, j + 1])
            cb = plt.colorbar(intensity, cax=cbaxes, label='Photons')

        for i in range(event_pix.shape[1]):
            axs_top[j].plot(event_pix[1, i] - crop[1].start, event_pix[0, i] - crop[0].start, marker='.', color='r')

        dif_wcs = l3.observation[0, j, event.location[0], event.location[1]].wcs.dropaxis(-1).dropaxis(-1)
        axs_bottom.append(fig1.add_subplot(spec1[1, j],
                                           projection=dif_wcs))

        dif_im = axs_bottom[j].imshow(dif[seq], origin='lower', vmin=-dif_thresh, vmax=dif_thresh)
        # dif_im = axs_bottom[j].imshow(dif[seq], origin='lower')
        axs_bottom[j].coords[0].set_ticklabel(exclude_overlapping=True)
        axs_bottom[j].coords[0].set_axislabel('Solar X (arcsec)')
        axs_bottom[j].coords[1].set_axislabel('Solar Y (arcsec)')
        if j > 0:
            axs_bottom[j].coords[1].set_ticklabel_visible(False)
        if j == len(seqs) - 1:
            cbaxes = fig1.add_subplot(spec1[1, j + 1])
            cb = plt.colorbar(dif_im, cax=cbaxes, label='Photons')

        shp = int_cutout.shape
        points = np.array([(0, 0), (0, shp[1]), (shp[0], shp[1]), (shp[0], 0)])
        l3_world_coords = int_wcs.all_pix2world(points, 0)
        dif_pix = dif_wcs.all_world2pix(l3_world_coords, 0)

        lw = 2
        box = patches.Polygon(dif_pix,
                              closed=True, edgecolor='r', facecolor='none', linewidth=lw,
                              )
        axs_bottom[j].add_patch(box)

    #
    # Begine Line Profile Fits at all event_pix

    fig2, axs = plt.subplots(n, n, constrained_layout=True, figsize=(7.1, 5))

    axs = axs[::-1]  # corrects subplots to match dot locations
    flat_axs = axs.flatten()
    colors = ['r', 'g', 'b', 'black']

    lp_max = l4.cube_list[brightest_pix[0]][brightest_pix[1], brightest_pix[2], :].max()

    for j, seq in enumerate(seqs):
        for i in range(event_pix.shape[1]):
            line_profile = np.array(l4.cube_list)[seq, event_pix[0, i], event_pix[1, i], :]
            domain = np.array(l4.velocity_axis)

            p0 = [line_profile.max(), line_profile.max() / 2, 50, 50, 100, -100]
            # p0 = [14, 4, 50, 50, -90, 110]

            bounds = (
                [0, 0, 15, 15, -200, -200], [line_profile.max() * 1.2, line_profile.max() * 1.2, 100, 100, 200, 200])
            fit_params, fit_cov = scipy.optimize.curve_fit(two_gauss, domain[guass_fit_trim: -guass_fit_trim],
                                                           line_profile[guass_fit_trim: -guass_fit_trim], p0=p0,
                                                           bounds=bounds)
            # fit_params = gauss_fit(domain[guass_fit_trim: -guass_fit_trim], line_profile[guass_fit_trim: -guass_fit_trim])

            flat_axs[i].plot(domain, line_profile, marker='.', linestyle='None', color=colors[j], ms=3)
            flat_axs[i].plot(domain, two_gauss(domain, *fit_params), color=colors[j], linewidth=1)
            flat_axs[i].set_ylim(top=lp_max)
            flat_axs[i].xaxis.set_minor_locator(AutoMinorLocator())
            w1 = '%.0f' % fit_params[2]
            w2 = '%.0f' % fit_params[3]
            vel1 = '%.0f' % fit_params[4]
            vel2 = '%.0f' % fit_params[5]
            flat_axs[i].annotate('v =' + vel1 + ', ' + vel2, (60, lp_max - 2.5 * (j + 1)), color=colors[j],
                                 fontsize=7)
            # flat_axs[i].annotate('v =' + vel1 + ', ' + vel2, (110, lp_max - 2 * (j+1)), color=colors[j],
            #                      fontsize=7)

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
    plt.subplots_adjust(top=.97, right=.97, left=.105)

    # Time series plots of intensity for each gaussian

    return fig1, fig2


def ee_deepdive_movie(event, seqs, event_pad, guass_fit_trim, dif_thresh, time_trim=None):
    l4 = level_4.Level_4.from_pickle(event.mart_inverted_pickle_path)

    if seqs is None:
        seqs = np.arange(len(l4.cube_list))
    if time_trim:
        seqs = seqs[time_trim]

    # plot = l4.plot()
    # plt.show()
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
    fig1 = plt.figure(figsize=(10, 10))
    outer_grid = fig1.add_gridspec(nrows=2, ncols=1)
    image_grid = outer_grid[0, 0].subgridspec(nrows=1, ncols=2, wspace=.5)
    lp_grid = outer_grid[1, 0].subgridspec(nrows=3, ncols=3)

    crop = (
        slice(brightest_pix[1] - event_pad, brightest_pix[1] + event_pad + 1),
        slice(brightest_pix[2] - event_pad, brightest_pix[2] + event_pad + 1))

    int_max = l4_int[:, crop[0], crop[1]].max()
    int_wcs = l4.wcs_list[0].dropaxis(0).slice((crop[0], crop[1]))
    dif_wcs = l3.observation[0, 0, event.location[0], event.location[1]].wcs.dropaxis(-1).dropaxis(-1)

    ax_invert = fig1.add_subplot(image_grid[0, 0], projection=int_wcs)
    int_cutout = l4_int[seqs[0], crop[0], crop[1]]
    intensity = ax_invert.imshow(int_cutout, vmax=int_max)
    t = ax_invert.annotate(times[seqs[0]].strftime('%H:%M:%S'), (.25, .25), color='w')
    t.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
    ax_invert.coords[0].set_ticklabel(exclude_overlapping=True)
    ax_invert.coords[0].set_axislabel('Solar X (arcsec)')
    ax_invert.coords[1].set_axislabel('Solar Y (arcsec)')
    ax_invert.set_title('Total Intensity')
    cb = plt.colorbar(intensity, ax=ax_invert, label='Photons')

    for i in range(event_pix.shape[1]):
        ax_invert.plot(event_pix[1, i] - crop[1].start, event_pix[0, i] - crop[0].start, marker='.', color='r')

    ax_dif = fig1.add_subplot(image_grid[0, 1], projection=dif_wcs)

    dif_im = ax_dif.imshow(dif[seqs[0]], origin='lower', vmin=-dif_thresh, vmax=dif_thresh)
    ax_dif.coords[0].set_ticklabel(exclude_overlapping=True)
    ax_dif.coords[0].set_axislabel('Solar X (arcsec)')
    ax_dif.coords[1].set_axislabel('Solar Y (arcsec)')
    ax_dif.set_title('Channel 2 - 3')
    cb = plt.colorbar(dif_im, ax=ax_dif, label='Photons')

    shp = int_cutout.shape
    points = np.array([(0, 0), (0, shp[1]), (shp[0], shp[1]), (shp[0], 0)])
    l3_world_coords = int_wcs.all_pix2world(points, 0)
    dif_pix = dif_wcs.all_world2pix(l3_world_coords, 0)

    lw = 2
    box = patches.Polygon(dif_pix,
                          closed=True, edgecolor='r', facecolor='none', linewidth=lw,
                          )
    ax_dif.add_patch(box)

    axs = lp_grid.subplots()
    axs = axs[::-1]  # corrects subplots to match dot locations
    flat_axs = axs.flatten()
    lp_max = l4.cube_list[brightest_pix[0]][brightest_pix[1], brightest_pix[2], :].max()
    lps = []
    red_lps = []
    blue_lps = []
    fit = []
    vels = []
    for i in range(event_pix.shape[1]):
        line_profile = np.array(l4.cube_list)[seqs[0], event_pix[0, i], event_pix[1, i], :]
        domain = np.array(l4.velocity_axis)

        p0 = [line_profile.max(), line_profile.max() / 2, 50, 50, 100, -100]
        # p0 = [14, 4, 50, 50, -90, 110]

        bounds = (
            [0, 0, 15, 15, -200, -200], [line_profile.max() * 1.2, line_profile.max() * 1.2, 100, 100, 200, 200])

        fit_params, fit_cov = scipy.optimize.curve_fit(two_gauss, domain[guass_fit_trim: -guass_fit_trim],
                                                       line_profile[guass_fit_trim: -guass_fit_trim], p0=p0,
                                                       bounds=bounds)
        # fit_params = gauss_fit(domain,line_profile)

        red_lps.append(
            flat_axs[i].plot(domain, one_gauss(domain, fit_params[0], fit_params[2], fit_params[4]), linestyle='--',
                             color='r'))
        blue_lps.append(
            flat_axs[i].plot(domain, one_gauss(domain, fit_params[1], fit_params[3], fit_params[5]), linestyle='--',
                             color='b'))
        fit.append(flat_axs[i].plot(domain, two_gauss(domain, *fit_params), color='fuchsia', linewidth=1.5))
        lps.append(flat_axs[i].plot(domain, line_profile, marker='.', linestyle='None', color='lime', ms=4))
        flat_axs[i].set_ylim(top=lp_max)
        vel1 = '%.0f' % fit_params[4]
        vel2 = '%.0f' % fit_params[5]
        vels.append(flat_axs[i].annotate('v =' + vel1 + ', ' + vel2, (130, lp_max - 1.5 - 1.75 * j), color='black',
                                         fontsize=7))
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

    def update(frame):
        intensity.set_data(l4_int[frame, crop[0], crop[1]])
        dif_im.set_data(dif[frame])
        t.set_text(times[frame].strftime('%H:%M:%S'))
        for i in range(event_pix.shape[1]):
            line_profile = np.array(l4.cube_list)[frame, event_pix[0, i], event_pix[1, i], :]
            domain = np.array(l4.velocity_axis)
            p0 = [line_profile.max(), line_profile.max() / 2, 50, 50, 100, -100]
            # p0 = [14, 4, 50, 50, -90, 110]

            bounds = (
                [0, 0, 15, 15, -200, -200], [line_profile.max() * 1.2, line_profile.max() * 1.2, 100, 100, 200, 200])

            fit_params, fit_cov = scipy.optimize.curve_fit(two_gauss, domain[guass_fit_trim: -guass_fit_trim],
                                                           line_profile[guass_fit_trim: -guass_fit_trim], p0=p0,
                                                           bounds=bounds)
            # fit_params = gauss_fit(domain, line_profile)
            red_lps[i][0].set_ydata(one_gauss(domain, fit_params[0], fit_params[2], fit_params[4]))
            blue_lps[i][0].set_ydata(one_gauss(domain, fit_params[1], fit_params[3], fit_params[5]))
            lps[i][0].set_ydata(line_profile)
            fit[i][0].set_ydata(two_gauss(domain, *fit_params))
            vel1 = '%.0f' % fit_params[4]
            vel2 = '%.0f' % fit_params[5]
            vels[i].set_text('v =' + vel1 + ', ' + vel2)

    return FuncAnimation(fig1, update, seqs)


def ee_deepdive_timeseries_plot(event, seqs, event_pad, guass_fit_trim, dif_thresh, time_trim=None):
    l4 = level_4.Level_4.from_pickle(event.mart_inverted_pickle_path)

    if seqs is None:
        seqs = np.arange(len(l4.cube_list))
    if time_trim:
        seqs = seqs[time_trim]

    # plot = l4.plot()
    # plt.show()
    l4_int = l4.integrated_intensity
    brightest_pix = np.unravel_index(l4_int.argmax(), l4_int.shape)

    l3 = level_3.Level_3.from_pickle(level_3.ov_final_path)
    times = l3.time
    event_imgs = l3.observation.data[..., event.location[0], event.location[1]]

    event_pix = []
    spread = 2

    n = 0
    for i in [-spread, 0, spread]:
        n += 1
        for j in [-spread, 0, spread]:
            event_pix.append([brightest_pix[1] + i, brightest_pix[2] + j])

    event_pix = np.array(event_pix).T
    double_gauss_fits = np.empty((seqs.shape[0], event_pix.shape[1], 6))

    for seq in seqs:
        for i in range(event_pix.shape[1]):
            line_profile = np.array(l4.cube_list)[seq, event_pix[0, i], event_pix[1, i], :]
            domain = np.array(l4.velocity_axis)

            p0 = [line_profile.max(), line_profile.max() / 2, 50, 50, 100, -100]
            # p0 = [14, 4, 50, 50, -90, 110]

            bounds = (
                [0, 0, 15, 15, -200, -200], [line_profile.max() * 1.2, line_profile.max() * 1.2, 100, 100, 200, 200])

            fit_params, fit_cov = scipy.optimize.curve_fit(two_gauss, domain[guass_fit_trim: -guass_fit_trim],
                                                           line_profile[guass_fit_trim: -guass_fit_trim], p0=p0,
                                                           bounds=bounds)

            double_gauss_fits[seq, i] = fit_params

    world_1 = l4.wcs_list[0].pixel_to_world_values(0, event_pix[1, 0], event_pix[0, 0])
    x1, y1 = world_1[1] * 3600, world_1[2] * 3600
    x1, y1 = "{:3.1f}".format(x1), "{:3.1f}".format(y1)
    world_2 = l4.wcs_list[0].pixel_to_world_values(0, event_pix[1, 5], event_pix[0, 5])
    x2, y2 = world_2[1] * 3600, world_2[2] * 3600
    x2, y2 = "{:3.1f}".format(x2), "{:3.1f}".format(y2)

    fig3, ax = plt.subplots(figsize=(7.1, 4))

    t = times[seqs].strftime('%H:%M:%S')
    ax.plot(t, double_gauss_fits[:, 0, 1], color='blue', label='x, y =' + str(x1) + '"' + ', ' + str(y1) + '"')
    ax.plot(t, double_gauss_fits[:, 0, 0], color='red', label='x, y =' + str(x1) + '"' + ', ' + str(y1) + '"')
    ax.plot(t, double_gauss_fits[:, 5, 1], color='blue', ls='--', label='x, y =' + str(x2) + '"' + ', ' + str(y2) + '"')
    ax.plot(t, double_gauss_fits[:, 5, 0], color='red', ls='--', label='x, y =' + str(x2) + '"' + ', ' + str(y2) + '"')
    ax.set_xlabel('UTC')
    ax.set_ylabel('Intensity (Photons)')
    ax.xaxis.set_major_locator(MultipleLocator(6))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend()

    # fig3, ax = plt.subplots()
    # # ax.plot(seqs, double_gauss_fits[:, 0, 5], color='blue')
    # ax.plot(seqs, double_gauss_fits[:, 5, 4], color='red')
    #
    # # print(l4.wcs_list[0])
    # position = l4.wcs_list[0].pixel_to_world(0,event_pix[1,5],event_pix[0,5])
    # print(position.value)


    return fig3


if __name__ == '__main__':
    # extent = 150
    # domain = np.arange(-extent, extent)
    # sig = 50
    # test = one_gauss(domain, 100, sig, -50)
    # test += one_gauss(domain,50,25,15)
    # # test += np.random.normal(0, .01, domain.shape[0])
    #
    # fit_param = gauss_fit(domain,test)
    # print(fit_param)
    # if fit_param[1] is None:
    #     fit = one_gauss(domain,fit_param[0],fit_param[2],fit_param[4])
    # else:
    #     fit = two_gauss(domain,*fit_param)
    #
    # fig,ax = plt.subplots()
    # ax.plot(domain, test, marker='.', linestyle='None', color='r')
    # ax.plot(domain, fit)
    # plt.show()
    #

    #
    # event = l3_events.perfectx
    # seqs = [6, 11, 15, 19] #for static figure
    # time_trim = slice(0, -4)

    event = l3_events.otherx
    seqs = [6, 10, 15, 18]
    time_trim = None

    ### Doesn't work when the thing you care about isn't the brightest in the frame
    # event = l3_events.big_blue
    # seqs = [4, 8, 13, 18]

    event_pad = 7
    guass_fit_trim = 7
    dif_thresh = 50

    # fig1, fig2 = ee_deepdive_figures(event, seqs, event_pad, guass_fit_trim, dif_thresh)
    #
    # filepath1 = event.name + '_inverta.pdf'
    # filepath2 = event.name + '_invertb.pdf'
    # fig1.savefig(fig_path / filepath1)
    # fig2.savefig(fig_path / filepath2)
    # # plt.show()
    #
    # seqs = None
    # movie = ee_deepdive_movie(event, seqs, event_pad, guass_fit_trim, dif_thresh, time_trim=time_trim)
    # movie_path = event.name + '_movie.mp4'
    # movie.save(fig_path / movie_path, 'ffmpeg', dpi=200)


    fig3 = ee_deepdive_timeseries_plot(event, None, event_pad, guass_fit_trim, dif_thresh, time_trim)
    filepath3 = event.name + '_timeseries.pdf'
    fig3.savefig(fig_path / filepath3)
