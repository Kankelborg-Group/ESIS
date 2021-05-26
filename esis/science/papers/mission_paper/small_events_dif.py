from esis.data import level_3
import matplotlib.pyplot as plt
from esis.science.papers.mission_paper import fig_path  # can't figure out relative import here
import matplotlib.patheffects as PathEffects
import matplotlib.patches as patches
from esis.flight import l3_events

plt.rcParams.update({'font.size': 9})

if __name__ == '__main__':

    l3 = level_3.Level_3.from_pickle(level_3.ov_final_path)
    times = l3.time
    img1 = l3.observation.data[:, 1]
    img2 = l3.observation.data[:, 2]
    dif = img1 - img2

    fig = plt.figure(
        figsize=(7.1,7),
                     # constrained_layout=True,
                     )


    events = [l3_events.big_blue, l3_events.little_red, l3_events.perfectx]
    label = ['a', 'b', 'c']
    seq = [17, 19, 12]
    axs = []
    for i, event, in enumerate(events):

        axs.append(fig.add_subplot(3, 3, 1+i*3,
                                   projection=l3.observation[seq[i], 0, event.location[0], event.location[1]].wcs.dropaxis(
                                       -1).dropaxis(-1),
                                   ))
        axs[i].imshow(dif[seq[i], event.location[0], event.location[1]])
        axs[i].set_ylabel('Solar Y (arcsec)')
        axs[i].set_xlabel('Solar X (arcsec)')
        axs[i].coords[0].set_ticks(number=4)
        axs[i].coords[1].set_ticks(number=4)
        axs[i].coords[0].display_minor_ticks(True)
        axs[i].coords[1].display_minor_ticks(True)
        num = axs[i].annotate(label[i] + ')', (1,35), color='w')
        num.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
        t = axs[i].annotate(times[seq[i]].strftime('%H:%M:%S'), (.125, .125), color='w', size = 8)
        t.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])

    for i, event, in enumerate(events):

        axs.append(fig.add_subplot(3, 3, 2+i*3,
                                   projection=l3.observation[seq[i], 0, event.location[0], event.location[1]].wcs.dropaxis(
                                       -1).dropaxis(-1),
                                   ))
        axs[i+3].imshow(img1[seq[i], event.location[0], event.location[1]])
        axs[i+3].set_ylabel('Solar Y (arcsec)')
        axs[i+3].set_xlabel('Solar X (arcsec)')
        axs[i+3].coords[0].set_ticks(number=4)
        # axs[i+3].coords[1].set_ticks(number=4)
        axs[i+3].coords[1].set_ticklabel_visible(False)
        axs[i+3].coords[0].display_minor_ticks(True)
        axs[i+3].coords[1].display_minor_ticks(True)
        # num = axs[i].annotate(label[i] + ')', (1,35), color='w')
        # num.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
        # t = axs[i].annotate(times[seq[i]], (.5, .5), color='w')
        # t.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])

    for i, event, in enumerate(events):

        axs.append(fig.add_subplot(3, 3, 3+i*3,
                                   projection=l3.observation[seq[i], 0, event.location[0], event.location[1]].wcs.dropaxis(
                                       -1).dropaxis(-1),
                                   ))
        axs[i+6].imshow(img2[seq[i], event.location[0], event.location[1]])
        axs[i+6].set_ylabel('Solar Y (arcsec)')
        axs[i+6].set_xlabel('Solar X (arcsec)')
        axs[i+6].coords[0].set_ticks(number=4)
        # axs[i+6].coords[1].set_ticks(number=4)
        axs[i + 6].coords[1].set_ticklabel_visible(False)
        axs[i+6].coords[0].display_minor_ticks(True)
        axs[i+6].coords[1].display_minor_ticks(True)
        # num = axs[i].annotate(label[i] + ')', (1,35), color='w')
        # num.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])
        # t = axs[i].annotate(times[seq[i]], (.5, .5), color='w')
        # t.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])

    axs[0].set_title('Channel 2-3')
    # axs[1].set_title('Channel 2-3')
    # axs[2].set_title('Channel 2-3')

    axs[3].set_title('Channel 2')
    # axs[4].set_title('Channel 2')
    # axs[5].set_title('Channel 2')

    axs[6].set_title('Channel 3')
    # axs[7].set_title('Channel 3')
    # axs[8].set_title('Channel 3')

    #


    #
    # fig.canvas.draw()
    # fig.tight_layout()
    # plt.subplot_tool()
    plt.subplots_adjust(hspace=.4,wspace=.1)
    plt.show()
    fig.savefig(fig_path/'dif_events.pdf')
