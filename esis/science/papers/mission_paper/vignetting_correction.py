from esis.data import level_3
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from esis.science.papers.mission_paper import fig_path  # can't figure out relative import here
import matplotlib.patheffects as PathEffects

plt.rcParams.update({'font.size': 9})
if __name__ == '__main__':
    sequence = 15
    scale = 15
    fit_window = [300, -300]
    l3_crop = 500

    l3_vig = level_3.Level_3.from_pickle(level_3.ov_Level3_masked)
    # means = l3_vig.masked_mean_normalization()
    # brightest_channel = 0
    # for i in l3_vig.lev1_cameras:
    #     l3_vig.observation.data[:, i, :, :] *= means[:, brightest_channel, :, :] / means[:, i, :, :]
    l3_vig = l3_vig.normalize_intensites()

    l3_fin = level_3.Level_3.from_pickle(level_3.ov_final_path)
    print(l3_fin.vignetting_correction_params)

    masked_imgs_vig = l3_vig.observation.data
    combined_mask = l3_fin.observation.mask[sequence, 1] * l3_fin.observation.mask[sequence, 2]
    masked_im1 = masked_imgs_vig[sequence, 1] * combined_mask
    masked_im2 = masked_imgs_vig[sequence, 2] * combined_mask

    dif1 = masked_im1 - masked_im2
    dif1_cp = np.copy(dif1)
    dif1_cp[dif1_cp == 0] = np.nan
    column_mean_before = np.nanmean(dif1_cp, axis=0)
    start, stop = (np.arange(column_mean_before.shape[0])[fit_window[0]],
                   np.arange(column_mean_before.shape[0])[fit_window[1]])
    column_mean_before = column_mean_before[fit_window[0]:fit_window[1]]
    poly_fit = np.polynomial.Polynomial.fit(np.arange(column_mean_before.shape[0]), column_mean_before, deg=1)
    slope1 = poly_fit.coef[1]
    bfit1 = slope1 * np.linspace(-1, 1, column_mean_before.shape[0]) + poly_fit.coef[0]

    masked_imgs = l3_fin.observation.data
    masked_im1 = masked_imgs[sequence, 1] * combined_mask
    masked_im2 = masked_imgs[sequence, 2] * combined_mask

    dif2 = masked_im1 - masked_im2

    dif2_cp = np.copy(dif2)
    dif2_cp[dif2_cp == 0] = np.nan
    column_mean_after = np.nanmean(dif2_cp, axis=0)
    column_mean_after = column_mean_after[fit_window[0]:fit_window[1]]
    poly_fit = np.polynomial.Polynomial.fit(np.arange(column_mean_after.shape[0]), column_mean_after, deg=1)
    slope2 = poly_fit.coef[1]
    bfit2 = slope2 * np.linspace(-1, 1, column_mean_after.shape[0]) + poly_fit.coef[0]

    letter_pos = (10,700)
    lw=1.1

    fig = plt.figure(figsize=[7.1, 5])
    ax1 = plt.subplot(2, 2, 1, projection=l3_fin.observation[sequence, 1, l3_crop:, :].wcs.dropaxis(-1).dropaxis(-1))
    ax1.imshow(dif1[..., l3_crop:, :], vmin=-scale, vmax=scale)
    ax1.axvline(start, color='r', label='Fit Region Edge')
    ax1.axvline(stop, color='r')
    ax1.set_ylabel('Solar Y (arcsec)')
    ax1.coords[0].set_ticklabel_visible(False)
    ax1.coords[0].set_ticks_visible(False)
    ax1.set_title('Masked Difference Image (Uncorrected)')
    a = ax1.annotate('a)', letter_pos, color='w')
    a.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])
    ax1.legend()

    ax2 = plt.subplot(2, 2, 3, projection=l3_fin.observation[sequence, 1, l3_crop:, :].wcs.dropaxis(-1).dropaxis(-1))
    ax2.imshow(dif2[..., l3_crop:, :], vmin=-scale, vmax=scale)
    ax2.axvline(start, color='r')
    ax2.axvline(stop, color='r')
    ax2.set_ylabel('Solar Y (arcsec)')
    ax2.set_xlabel('Solar X (arcsec)')
    ax2.set_title('Masked Difference Image (Corrected)')
    c = ax2.annotate('c)', letter_pos, color='w')
    c.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])

    ax3 = plt.subplot(2, 2, 2)
    ax3.plot(column_mean_before)
    ax3.plot(bfit1, color='r', label='slope = ' + str(np.round(slope1, 2)))
    # ax3.legend()
    ax3.set_xticklabels([])
    ax3.set_xticks([])
    ax3.set_ylabel('Column Mean (Photons)')
    ax3.set_title('Linear Fit (Uncorrected)')
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    b = ax3.annotate('b)', (0,3.5), color='black')
    # b.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(column_mean_after)
    ax4.plot(bfit2, color='r', label='slope = ' + str(np.round(slope2, 2)))
    # ax4.legend()
    ax4.set_ylim(ax3.get_ylim())
    ax4.set_ylabel('Column Mean (Photons)')
    ax4.set_xlabel('Column (pix)')
    ax4.set_title('Linear Fit (Corrected)')
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    d = ax4.annotate('d)', (0, 3.5), color='black')

    fig.savefig(fig_path / 'vig_correct.pdf')
    plt.show()
