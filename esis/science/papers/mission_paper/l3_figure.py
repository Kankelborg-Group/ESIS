from esis.data import level_3
import matplotlib.pyplot as plt
from esis.science.papers.mission_paper import fig_path #can't figure out relative import here
from . import fig_path
if __name__ == '__main__':

    seq = 15  #find apogee image maybe
    scale = 30

    l3 = level_3.Level3.from_pickle(level_3.ov_final_path)
    times = l3.time
    fig, ax = plt.subplots(1,2, sharex=True, sharey=True,
                           subplot_kw=dict(projection=l3.observation[seq,1].wcs.dropaxis(-1).dropaxis(-1)),
                           figsize = [12,5]
                           )
    img1 = l3.observation.data[seq, 1]
    img2 = l3.observation.data[seq, 2]
    ax[0].imshow(img1)
    ax[1].imshow(img1-img2, vmax=scale, vmin=-scale)
    ax[0].set_xlabel('Solar X')
    ax[1].set_xlabel('Solar X')
    ax[0].set_ylabel('Solar Y')
    ax[1].set_ylabel('Solar Y')
    t = ax[0].annotate(times[seq],(3,3),color = 'w')

    fig.savefig(fig_path/'l3.pdf')
