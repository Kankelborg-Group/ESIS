from esis.data import level_3
import matplotlib.pyplot as plt
from esis.science.papers.mission_paper import fig_path  # can't figure out relative import here
import matplotlib.patheffects as PathEffects
import matplotlib.patches as patches
from esis.flight import l3_events
from kgpy.plot import CubeSlicer
import numpy as np
from matplotlib.animation import FuncAnimation
from kgpy.observatories.sdo import aia
import astropy.units as u
from matplotlib.colors import PowerNorm

# change labels to e1, e2, e3
if __name__ == '__main__':
    l3 = level_3.Level_3.from_pickle(level_3.ov_final_path)

    times = l3.time
    main_event = l3_events.main_event

    event = l3.observation.data[..., main_event.location[0], main_event.location[1]]
    # event = l3.observation.data
    dif = event[:, 1, ...] - event[:, 2, ...]

    # slicer = CubeSlicer(dif, origin='lower',vmax = np.percentile(dif,99.5))
    # plt.show()

    fig, axs = plt.subplots(3, 3,
                            figsize=(7.1, 7.1),
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
        t = axs[i, 0].annotate(times[seq].strftime('%H:%M:%S'), (1, 1), color='w', size=8)
        t.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])

        loc_a = axs[i, 0].annotate('$\longleftarrow$', (62, 15), color='w', rotation=22.5 - 45)
        loc_a.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
        a = axs[i, 0].annotate('e1)', (77, 15), color='w', rotation=0)
        a.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])

        loc_b = axs[i, 0].annotate('$\longleftarrow$', (62, 50), color='w', rotation=22.5 - 45)
        loc_b.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
        b = axs[i, 0].annotate('e2)', (77, 50), color='w', rotation=0)
        b.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])

        loc_c = axs[i, 0].annotate('$\longleftarrow$', (12, 45), color='w', rotation=180)
        loc_c.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
        c = axs[i, 0].annotate('e3)', (1, 45), color='w', rotation=0)
        c.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])

        axs[i, 1].imshow(event[seq, 1])
        axs[i, 1].coords[0].set_ticklabel_visible(False)
        axs[i, 1].coords[1].set_ticklabel_visible(False)

        axs[i, 2].imshow(event[seq, 2])
        axs[i, 2].coords[0].set_ticklabel_visible(False)
        axs[i, 2].coords[1].set_ticklabel_visible(False)

        if i == 0:
            axs[i, 0].set_title('Channel 2-3')
            axs[i, 1].set_title('Channel 2')
            axs[i, 2].set_title('Channel 3')

        if i == 2:
            for j in range(len(seqs)):
                axs[i, j].coords[0].set_ticklabel_visible(True)
                axs[i, j].coords[0].set_axislabel('Solar X (arcsec)')

        for j in range(len(seqs)):
            axs[i, j].coords[0].display_minor_ticks(True)
            axs[i, j].coords[1].display_minor_ticks(True)

    # plt.subplot_tool()
    plt.subplots_adjust(right=.95, top=.95)
    fig.savefig(fig_path / 'main_event_dif.pdf')
    # Still AIA Context
    s = seqs[1]-1

    aia_304_obj = l3.to_aia_object()
    aia_171_obj = l3.to_aia_object(aia_channel=171 * u.AA)
    aia_193_obj = l3.to_aia_object(aia_channel=193 * u.AA)

    thresh_304 = np.percentile(aia_304_obj.observation.data[s, 0, main_event.location[0], main_event.location[1]], 99)
    thresh_171 = np.percentile(aia_171_obj.observation.data[s, 0, main_event.location[0], main_event.location[1]], 99.5)
    thresh_193 = np.percentile(aia_193_obj.observation.data[s, 0, main_event.location[0], main_event.location[1]], 99.5)


    fig3, axs = plt.subplots(3, 1,
                             figsize=(3.5,7.1),
                             subplot_kw=dict(projection=l3.observation[
                                 s, 0, main_event.location[0], main_event.location[1]].wcs.dropaxis(-1).dropaxis(-1)),
                             )

    aia_304 = axs[0].imshow(aia_304_obj.observation.data[s, 0, main_event.location[0], main_event.location[1]],
                            cmap='sdoaia304', vmax=thresh_304)
    aia_304_t = axs[0].annotate(aia_304_obj.observation.meta['times'][s].strftime('%H:%M:%S'), (1, 1), color='w',
                                size=8)
    aia_304_t.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
    axs[0].coords[0].set_ticklabel_visible(False)
    axs[0].coords[1].set_axislabel('Solar Y (arcsec)')

    aia_171 = axs[1].imshow(aia_171_obj.observation.data[s, 0, main_event.location[0], main_event.location[1]],
                            cmap='sdoaia171', vmax=thresh_171)
    aia_171_t = axs[1].annotate(aia_171_obj.observation.meta['times'][s].strftime('%H:%M:%S'), (1, 1), color='w',
                                size=8)
    aia_171_t.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
    axs[1].coords[0].set_ticklabel_visible(False)
    axs[1].coords[1].set_axislabel('Solar Y (arcsec)')

    aia_193 = axs[2].imshow(aia_193_obj.observation.data[s, 0, main_event.location[0], main_event.location[1]],
                            cmap='sdoaia193', vmax=thresh_193)
    aia_193_t = axs[2].annotate(aia_193_obj.observation.meta['times'][s].strftime('%H:%M:%S'), (1, 1), color='w',
                                size=8)
    aia_193_t.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
    axs[2].coords[0].set_axislabel('Solar X (arcsec)')
    axs[2].coords[1].set_axislabel('Solar Y (arcsec)')


    axs[0].set_title('AIA 304$\,\AA$')
    axs[1].set_title('AIA 171$\,\AA$')
    axs[2].set_title('AIA 193$\,\AA$')

    for j in range(axs.shape[0]):
        axs[j].coords[0].display_minor_ticks(True)
        axs[j].coords[1].display_minor_ticks(True)

        loc_a = axs[j].annotate('$\longleftarrow$', (62, 15), color='w', rotation=22.5 - 45)
        loc_a.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
        a = axs[j].annotate('e1)', (77, 15), color='w', rotation=0)
        a.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])

        loc_b = axs[j].annotate('$\longleftarrow$', (62, 50), color='w', rotation=22.5 - 45)
        loc_b.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
        b = axs[j].annotate('e2)', (77, 50), color='w', rotation=0)
        b.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])

        loc_c = axs[j].annotate('$\longleftarrow$', (12, 45), color='w', rotation=180)
        loc_c.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
        c = axs[j].annotate('e3)', (1, 45), color='w', rotation=0)
        c.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])

    plt.subplots_adjust(top=.91)
    fig3.savefig(fig_path / 'main_event_aia_context.pdf')



    # Animation
    fig2, axs = plt.subplots(2, 3,
                             figsize=(7.1, 6),
                             subplot_kw=dict(projection=l3.observation[
                                 0, 0, main_event.location[0], main_event.location[1]].wcs.dropaxis(-1).dropaxis(-1)),
                             )
    img_thresh = np.percentile(event[:, 1:2], 99.9)

    axs = axs.flatten()

    d = axs[0].imshow(dif[0], vmin=-scale, vmax=scale)
    axs[0].coords[0].set_axislabel('Solar X (arcsec)')
    axs[0].coords[1].set_axislabel('Solar Y (arcsec)')
    t = axs[0].annotate(times[0].strftime('%H:%M:%S'), (1, 1), color='w', size=8)
    t.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])

    loc_a = axs[0].annotate('$\longleftarrow$', (62, 15), color='w', rotation=22.5 - 45)
    loc_a.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
    a = axs[0].annotate('e1)', (77, 15), color='w', rotation=0)
    a.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])

    loc_b = axs[0].annotate('$\longleftarrow$', (62, 50), color='w', rotation=22.5 - 45)
    loc_b.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
    b = axs[0].annotate('e2)', (77, 50), color='w', rotation=0)
    b.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])

    loc_c = axs[0].annotate('$\longleftarrow$', (12, 45), color='w', rotation=180)
    loc_c.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
    c = axs[0].annotate('e3)', (1, 45), color='w', rotation=0)
    c.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])

    ch2 = axs[1].imshow(event[0, 1], vmax=img_thresh)
    axs[1].coords[0].set_axislabel('Solar X (arcsec)')
    axs[1].coords[1].set_ticklabel_visible(False)

    ch3 = axs[2].imshow(event[0, 2], vmax=img_thresh)
    axs[2].coords[0].set_axislabel('Solar X (arcsec)')
    axs[2].coords[1].set_ticklabel_visible(False)

    axs[0].set_title('Channel 2-3')
    axs[1].set_title('Channel 2')
    axs[2].set_title('Channel 3')

    thresh_304 = np.percentile(aia_304_obj.observation.data[:, 0, main_event.location[0], main_event.location[1]], 99.5)
    thresh_171 = np.percentile(aia_171_obj.observation.data[:, 0, main_event.location[0], main_event.location[1]], 99.5)
    thresh_193 = np.percentile(aia_193_obj.observation.data[:, 0, main_event.location[0], main_event.location[1]], 99.5)

    aia_304 = axs[3].imshow(aia_304_obj.observation.data[0, 0, main_event.location[0], main_event.location[1]],
                            cmap='sdoaia304', vmax=thresh_304)
    aia_304_t = axs[3].annotate(aia_304_obj.observation.meta['times'][0].strftime('%H:%M:%S'), (1, 1), color='w',
                                size=8)
    aia_304_t.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
    axs[3].coords[0].set_axislabel('Solar X (arcsec)')
    axs[3].coords[1].set_axislabel('Solar Y (arcsec)')

    aia_171 = axs[4].imshow(aia_171_obj.observation.data[0, 0, main_event.location[0], main_event.location[1]],
                            cmap='sdoaia171', vmax=thresh_171)
    aia_171_t = axs[4].annotate(aia_171_obj.observation.meta['times'][0].strftime('%H:%M:%S'), (1, 1), color='w',
                                size=8)
    aia_171_t.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
    axs[4].coords[0].set_axislabel('Solar X (arcsec)')
    axs[4].coords[1].set_ticklabel_visible(False)

    aia_193 = axs[5].imshow(aia_193_obj.observation.data[0, 0, main_event.location[0], main_event.location[1]],
                            cmap='sdoaia193', vmax=thresh_193)
    aia_193_t = axs[5].annotate(aia_193_obj.observation.meta['times'][0].strftime('%H:%M:%S'), (1, 1), color='w',
                                size=8)
    aia_193_t.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
    axs[5].coords[0].set_axislabel('Solar X (arcsec)')
    axs[5].coords[1].set_ticklabel_visible(False)

    axs[3].set_title('AIA 304$\,\AA$')
    axs[4].set_title('AIA 171$\,\AA$')
    axs[5].set_title('AIA 193$\,\AA$')

    for j in range(axs.shape[0]):
        axs[j].coords[0].display_minor_ticks(True)
        axs[j].coords[1].display_minor_ticks(True)


    def update(frame):
        d.set_data(dif[frame])
        ch2.set_data(event[frame, 1])
        ch3.set_data(event[frame, 2])
        t.set_text(times[frame].strftime('%H:%M:%S'))
        aia_304.set_data(aia_304_obj.observation.data[frame, 0, main_event.location[0], main_event.location[1]])
        aia_304_t.set_text(aia_304_obj.observation.meta['times'][frame].strftime('%H:%M:%S'))
        aia_171.set_data(aia_171_obj.observation.data[frame, 0, main_event.location[0], main_event.location[1]])
        aia_171_t.set_text(aia_171_obj.observation.meta['times'][frame].strftime('%H:%M:%S'))
        aia_193.set_data(aia_193_obj.observation.data[frame, 0, main_event.location[0], main_event.location[1]])
        aia_193_t.set_text(aia_193_obj.observation.meta['times'][frame].strftime('%H:%M:%S'))


    frames = np.arange(dif.shape[0])
    ani = FuncAnimation(fig2, update, frames)

    ani.save(fig_path / 'main_event_dif_movie.mp4', 'ffmpeg', dpi=200)

    plt.show()
