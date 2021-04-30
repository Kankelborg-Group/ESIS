from esis.data import level_3
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from esis.science.papers.mission_paper import fig_path  # can't figure out relative import here
import matplotlib.patheffects as PathEffects
from matplotlib.patches import Polygon

plt.rcParams.update({'font.size': 9})

if __name__ == '__main__':
    lev3 = level_3.Level_3.from_pickle(level_3.ov_final_path)

    imgs = lev3.observation.data
    imgs[imgs<0] = 1e-6
    sequence = 15
    p = 99.5
    lw = 1.1
    letter_pos = (10, 1190)

    wcs = lev3.observation.wcs
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True,
                            figsize=(7,7),
                            subplot_kw={'projection': wcs[sequence, 0]},
                            # constrained_layout=True,
                            )

    y_label_pad = 1
    x_label_pad = 1

    # figure a
    ax1 = axs[0, 0]
    x = ax1.coords[0]
    y = ax1.coords[1]
    ax1.imshow(imgs[sequence, 0], origin='lower',
               # vmax=np.percentile(imgs[sequence, 0], p),
               norm = colors.LogNorm(5,vmax = np.percentile(imgs[sequence, 0],99)),
               )
    ax1.set_title('Level-3 Channel 1')
    y.set_axislabel('Solar Y (arcsec)', minpad=y_label_pad)
    x.set_ticklabel_visible(False)

    a = ax1.annotate('a)', letter_pos, color='w')
    a.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])

    points = [(500, 840), (675, 360), (508, 0), (0, 0), (0, 980), (50, 1000)]
    poly = Polygon(points, closed=True, fill=False, hatch='x', color='red')
    ax1.add_patch(poly)

    disperse = ax1.annotate('$\quad\lambda$ \n $\longrightarrow$', (120, 1060), color='w', rotation=22.5)
    disperse.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])

    # figure b
    ax2 = axs[0, 1]
    x = ax2.coords[0]
    y = ax2.coords[1]
    ax2.imshow(imgs[sequence, 1], origin='lower',
               # vmax=np.percentile(imgs[sequence, 0], p),
               norm = colors.LogNorm(5,vmax = np.percentile(imgs[sequence, 1],99)))
    ax2.set_title('Level-3 Channel 2')
    x.set_ticklabel_visible(False)
    y.set_ticklabel_visible(False)

    b = ax2.annotate('b)', letter_pos, color='w')
    b.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])

    points = [(0, 0), (0, 490), (430, 700), (850, 525), (1020, 32), (1010, 0)]
    poly = Polygon(points, closed=True, fill=False, hatch='x', color='red')
    ax2.add_patch(poly)

    disperse = ax2.annotate('$\quad\lambda$ \n $\longrightarrow$', (980, 20), color='w', rotation=22.5 + 45)
    disperse.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])



    # figure c
    ax3 = axs[1, 0]
    x = ax3.coords[0]
    y = ax3.coords[1]
    ax3.imshow(imgs[sequence, 2], origin='lower',
               # vmax=np.percentile(imgs[sequence, 1], p),
               norm = colors.LogNorm(5,vmax = np.percentile(imgs[sequence, 2],99)))
    ax3.set_title('Level-3 Channel 3')
    x.set_axislabel('Solar X (arcsec)', minpad=x_label_pad)
    y.set_axislabel('Solar Y (arcsec)', minpad=y_label_pad)

    c = ax3.annotate('c)', letter_pos, color='w')
    c.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])

    points = [(1270, 0), (280, 0), (270, 30), (405, 455), (910, 662), (1270, 505)]
    poly = Polygon(points, closed=True, fill=False, hatch='x', color='red')
    ax3.add_patch(poly)

    disperse = ax3.annotate('$\quad\lambda$ \n $\longleftarrow$', (90, 50), color='w', rotation=22.5 - 90)
    disperse.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])

    # figure d
    ax4 = axs[1, 1]
    x = ax4.coords[0]
    y = ax4.coords[1]
    ax4.imshow(imgs[sequence, 3], origin='lower',
               # vmax=np.percentile(imgs[sequence, 1], p),
               norm = colors.LogNorm(5,vmax = np.percentile(imgs[sequence, 3],99)))
    ax4.set_title('Level-3 Channel 4')
    x.set_axislabel('Solar X (arcsec)',minpad=x_label_pad)
    y.set_axislabel_position('r')
    y.set_ticklabel_visible(False)

    d = ax4.annotate('d)', letter_pos, color='w')
    d.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])

    points = [(1270, 0), (765, 0), (592, 422), (767, 855), (1255, 1000), (1270, 985)]
    poly = Polygon(points, closed=True, fill=False, hatch='x', color='red')
    ax4.add_patch(poly)

    disperse = ax4.annotate('$\quad\lambda$ \n $\longleftarrow$', (1060, 1060), color='w', rotation=22.5 - 45)
    disperse.set_path_effects([PathEffects.withStroke(linewidth=lw, foreground='black')])

#supposed workaround did NOT work for me.
    # fig.canvas.draw()
    # fig.tight_layout()
    plt.subplots_adjust(right=.955, wspace=0, hspace=.15, top = .925)
    # plt.subplot_tool()
    fig.savefig(fig_path / 'mgx_overlap.pdf')
    plt.show()
