from esis.data import level_3
import matplotlib.pyplot as plt
from esis.science.papers.mission_paper import fig_path #can't figure out relative import here
import matplotlib.patheffects as PathEffects
import matplotlib.patches as patches
from esis.flight import l3_events


plt.rcParams.update({'font.size': 9})

if __name__ == '__main__':

    seq = 16  #find apogee image maybe


    l3 = level_3.Level_3.from_pickle(level_3.ov_final_path)
    times = l3.time
    img1 = l3.observation.data[seq, 1]
    img2 = l3.observation.data[seq, 2]
    dif = img1-img2





    fig, ax = plt.subplots(figsize=(7.5,7.5),
        subplot_kw=dict(projection=l3.observation[seq,1].wcs.dropaxis(-1).dropaxis(-1)),
                           )

    lw=1.1
    scale = 40

    ax.imshow(dif, origin='lower', vmin=-scale, vmax=scale)
    ax.set_title('Level-3 Difference (Camera 2 - Camera 3)')
    ax.set_xlabel('Solar X (arcsec)')
    ax.set_ylabel('Solar Y (arcsec)')
    ax.coords[0].display_minor_ticks(True)
    ax.coords[1].display_minor_ticks(True)
    t = ax.annotate(times[seq],(5,5), color='w')
    t.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])

    events = [l3_events.big_blue,  l3_events.little_red, l3_events.perfectx, l3_events.main_event]
    labels = ['a', 'b', 'c', 'd']
    for i,event in enumerate(events):
        ax.add_patch(event.rectangle)
        num = ax.annotate(labels[i]+')', (event.location[1].start-30, event.location[0].start-30), color='w')
        num.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])







    fig.savefig(fig_path/'l3_dif.pdf')
    plt.show()
