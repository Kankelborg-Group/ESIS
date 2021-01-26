from esis.data import level_3
import matplotlib.pyplot as plt
from esis.science.papers.mission_paper import fig_path #can't figure out relative import here

plt.rcParams.update({'font.size': 9})

if __name__ == '__main__':

    seq = 15  #find apogee image maybe
    scale = 30

    l3 = level_3.Level_3.from_pickle(level_3.ov_final_path)
    times = l3.time
    fig, ax = plt.subplots(1,2, sharex=True, sharey=True,
                           subplot_kw=dict(projection=l3.observation[seq,1].wcs.dropaxis(-1).dropaxis(-1)),
                           figsize=[7,4],
                           )
    img1 = l3.observation.data[seq, 1]
    img2 = l3.observation.data[seq, 2]

    # figure a)
    ax[0].imshow(img1, origin='lower')
    ax[0].set_xlabel('Solar X (arcsec)')
    ax[0].set_ylabel('Solar Y (arcsec)')
    t = ax[0].annotate(times[seq],(3,3), color='w')
    a = ax[0].annotate('a)', (3,1200), color='w')
    print(times[seq])


    #figure b)
    ax[1].imshow(img1-img2, origin='lower', vmax=scale, vmin=-scale)
    ax[1].set_xlabel('Solar X (arcsec)')
    b = ax[1].annotate('b)', (3,1200), color='w')
    ax[1].coords[1].set_ticklabel_visible(False)
    ax[1].coords[1].set_ticks_visible(False)


    fig.savefig(fig_path/'l3.pdf')
    # plt.show()
