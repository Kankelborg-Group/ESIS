from esis.data import level_3
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from esis.science.papers.mission_paper import fig_path #can't figure out relative import here
import matplotlib.patheffects as PathEffects


plt.rcParams.update({'font.size': 9})



if __name__ == '__main__':


    seq = 15
    cam = 1

    l3 = level_3.Level_3.from_pickle(level_3.ov_final_path)
    aia_304 = l3.to_aia_object()



    times = l3.time
    aia_times = aia_304.observation.meta['times']

    #start fig
    lw = 1.1

    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True,
                           subplot_kw=dict(projection=l3.observation[seq, 1].wcs.dropaxis(-1).dropaxis(-1)),
                           figsize=[7, 4],
                           )

    img1 = l3.observation.data[seq, cam]
    img2 = aia_304.observation.data[seq, cam]

    # figure a)
    ax[0].imshow(img1, origin='lower', vmax = np.percentile(img1,99.5))
    ax[0].set_xlabel('Solar X (arcsec)')
    ax[0].set_ylabel('Solar Y (arcsec)')
    ax[0].set_title('ESIS Level-3 O V 630 $\AA$')

    t1 = ax[0].annotate(times[seq], (5, 5), color='w')
    t1.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])

    a = ax[0].annotate('a)', (3, 1200), color='w')
    a.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])


    # figure b)
    ax[1].imshow(img2, origin='lower',cmap = 'sdoaia304', vmax = np.percentile(img2,99.5))
    ax[1].set_xlabel('Solar X (arcsec)')
    ax[1].coords[1].set_ticklabel_visible(False)
    # ax[1].coords[1].set_ticks_visible(False)
    ax[1].set_title('SDO AIA He II 304 $\AA$')

    t2 = ax[1].annotate(aia_times[seq].to_value('isot'), (5, 5), color='w')
    t2.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])

    b = ax[1].annotate('b)', (3,1200), color='w')
    b.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])


    fig.savefig(fig_path/'aia_coalign.pdf')
    # plt.show()
