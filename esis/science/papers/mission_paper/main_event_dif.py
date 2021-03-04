from esis.data import level_3
import matplotlib.pyplot as plt
from esis.science.papers.mission_paper import fig_path  # can't figure out relative import here
import matplotlib.patheffects as PathEffects
import matplotlib.patches as patches
from esis.flight import l3_events
from kgpy.plot import CubeSlicer
import numpy as np

if __name__ == '__main__':
    l3 = level_3.Level_3.from_pickle(level_3.ov_final_path_spikes)
    times = l3.time
    main_event = l3_events.main_event


    event = l3.observation.data[..., main_event.location[0], main_event.location[1]]
    dif = event[:, 1, ...] - event[:, 2, ...]


    fig, axs = plt.subplots(3, 3,
                            figsize=(7.5, 7.5),
                            subplot_kw=dict(projection=l3.observation[
                                0, 0, main_event.location[0], main_event.location[1]].wcs.dropaxis(-1).dropaxis(-1)),
                            )

    scale = 50
    ypad = 0
    seqs = [2, 12, 19]
    for i, seq in enumerate(seqs):
        axs[i, 0].imshow(dif[seq], vmin=-scale, vmax=scale)

        axs[i, 0].coords[0].set_ticklabel_visible(False)
        axs[i, 0].coords[1].set_axislabel('Solar Y (arcsec)', minpad=ypad)
        t = axs[i, 0].annotate(times[seq], (.125, .125), color='w', size=8)
        t.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])

        axs[i, 1].imshow(event[seq, 1])
        axs[i, 1].coords[0].set_ticklabel_visible(False)
        axs[i, 1].coords[1].set_ticklabel_visible(False)

        axs[i, 2].imshow(event[seq, 2])
        axs[i, 2].coords[0].set_ticklabel_visible(False)
        axs[i, 2].coords[1].set_ticklabel_visible(False)

        if i == 0:
            axs[i, 0].set_title('Camera 2-3')
            axs[i, 1].set_title('Camera 2')
            axs[i, 2].set_title('Camera 3')

        if i == 2:
            for j in range(len(seqs)):
                axs[i, j].coords[0].set_ticklabel_visible(True)
                axs[i, j].coords[0].set_axislabel('Solar X (arcsec)')

    # plt.subplot_tool()
    plt.subplots_adjust(right=.95, top=.95)
    plt.show()
    fig.savefig(fig_path / 'main_event.pdf')
