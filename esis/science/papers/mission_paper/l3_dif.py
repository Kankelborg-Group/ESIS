from esis.data import level_3
import matplotlib.pyplot as plt
from esis.science.papers.mission_paper import fig_path #can't figure out relative import here
import matplotlib.patheffects as PathEffects
import matplotlib.patches as patches
from esis.flight import l3_events
from matplotlib.animation import FuncAnimation
import numpy as np

from kgpy.plot import CubeSlicer


plt.rcParams.update({'font.size': 9})

if __name__ == '__main__':

    seq = 16  #find apogee image maybe


    l3 = level_3.Level_3.from_pickle(level_3.ov_final_path)
    times = l3.time
    img1 = l3.observation.data[:, 1]
    img2 = l3.observation.data[:, 2]
    dif = img1-img2

    fig, ax = plt.subplots(figsize=(7.1,7.1),
        subplot_kw=dict(projection=l3.observation[seq,1].wcs.dropaxis(-1).dropaxis(-1)),
                           )

    lw=1.1
    scale = 40

    d = ax.imshow(dif[16], origin='lower', vmin=-scale, vmax=scale)
    ax.set_title('Level-3 Difference (Channel 2 - Channel 3)')
    ax.set_xlabel('Solar X (arcsec)')
    ax.set_ylabel('Solar Y (arcsec)')
    ax.coords[0].display_minor_ticks(True)
    ax.coords[1].display_minor_ticks(True)
    t = ax.annotate(times[seq],(5,5), color='w')
    t.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])

    disperse2 = ax.annotate('$\longleftarrow$', (950, 1200), color='w', rotation=22.5-90-45, ha='center', va='center')
    disperse2.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])
    disperse2_txt = ax.annotate('Ch2', (950, 1150), color='w', ha='center', va='center')
    disperse2_txt.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])

    disperse3 = ax.annotate('$\longleftarrow$', (320, 1200), color='w', rotation=22.5-90, ha='center', va='center')
    disperse3.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])
    disperse3_txt = ax.annotate('Ch3', (350, 1150), color='w', ha='center', va='center')
    disperse3_txt.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])

    events = [l3_events.big_blue,  l3_events.little_red, l3_events.perfectx,l3_events.otherx, l3_events.main_event]
    labels = ['a', 'b', 'c', 'd', 'e']
    for i,event in enumerate(events):
        ax.add_patch(event.rectangle)
        num = ax.annotate(labels[i]+')', (event.location[1].start-30, event.location[0].start-30), color='w')
        num.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])

    fig.savefig(fig_path/'l3_dif.pdf')

    def update(frame):
        d.set_data(dif[frame])
        t.set_text(times[frame])



    frames = np.arange(dif.shape[0]-1)
    ani = FuncAnimation(fig, update, frames)

    ani.save(fig_path / 'l3_dif_movie.mp4', 'ffmpeg', dpi=200)
    plt.show()
