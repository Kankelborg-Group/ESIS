import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import astropy.units as u
from kgpy.observatories.sdo import aia
from esis.data import level_3
from esis.science.papers.mission_paper import fig_path  # can't figure out relative import here
import matplotlib.patheffects as PathEffects


if __name__ == '__main__':
    seq = 12

    lev3 = level_3.Level_3.from_pickle(level_3.ov_final_path)

    #estimate rough roll and pointing
    # l3_img = lev3.observation[15,1].data
    # l3_wcs = lev3.observation[15,1].wcs.dropaxis(-1).dropaxis(-1)
    # fig, ax = plt.subplots(figsize = [7.1,7.1],subplot_kw=dict(projection=l3_wcs))
    # ax.imshow(l3_img,vmax = np.percentile(l3_img, 99))
    # plt.show()

    aia_304 = aia.AIA.from_time_range(lev3.time[0] - 20 * u.s, lev3.time[-1] + 20 * u.s, channels=[304 * u.AA])
    print(lev3.time[seq])
    td = aia_304.time[:, 0, ...] - lev3.time[seq]  # should be the same for every camera
    aia_seq = np.abs(td.sec).argmin()

    slice1 = slice(300, 3800)
    slice2 = slice(300, 3750)
    # slice1 = slice(None)
    # slice2 = slice(None)
    pos = (slice1, slice2)

    aia_wcs = aia_304.wcs[aia_seq, 0].slice(pos)
    fig, ax = plt.subplots(figsize=(3.5,3.5),
        subplot_kw=dict(projection=aia_wcs)
    )
    aia_304_img = aia_304.intensity[aia_seq, 0, pos[0], pos[1]].value
    ax.imshow(aia_304_img, vmax = np.percentile(aia_304_img, 99.9), cmap = 'sdoaia304')

    ax.set_xlabel('Solar X (arcsec)')
    ax.coords[1].set_axislabel('Solar Y (arcsec)', minpad=-.2)
    aia_time = aia_304.time[aia_seq, 0]
    aia_time.format = 'isot'
    t = ax.annotate(aia_time, (5, 5), color='w', size=8)
    t.set_path_effects([PathEffects.withStroke(linewidth=1.1, foreground='black')])

    points = np.array([(629,11),(1077,185),(1267,630),(1081,1070),(644,1255),(197,1075),(17,633),(190,200)])
    wcs = lev3.observation[seq, 1].wcs.dropaxis(-1).dropaxis(-1)
    l3_world_coords = wcs.all_pix2world(points, 0)
    aia_pix = aia_wcs.all_world2pix(l3_world_coords, 0)

    lw = 2
    octagon = patches.Polygon(aia_pix,
                              closed=True, edgecolor='g', facecolor='none', linewidth=lw,
                              )
    ax.add_patch(octagon)

    # plt.subplot_tool()
    plt.subplots_adjust(left = .2, top = 1, bottom=.1)
    plt.show()
    fig.savefig(fig_path / 'esis_pointing.pdf')



