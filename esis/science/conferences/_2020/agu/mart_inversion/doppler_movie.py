from esis.data import Level_4
import pathlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import kgpy.moment
from esis.flight import l3_events


if __name__ == '__main__':

    event = l3_events.perfectx
    pickle_path = 'lev4_' + event.name +'_mart.pickle'
    html_path = event.name + '_doppler.html'


    path = pathlib.Path(__file__).parents[5] / 'flight/' / pickle_path

    event = (slice(None),slice(None))


    lev4 = Level_4.from_pickle(path)

    int = lev4.integrated_intensity
    int_wcs = lev4.int_wcs

    shifts = []
    for i,cube in enumerate(lev4.cube_list):
        threshold = np.percentile(int,95)

        shift = np.squeeze(kgpy.moment.percentile.shift(cube))
        # shift = cube.argmax(axis = -1)

        shift -= 21

        shift[int[i] < threshold] = 0
        shifts.append(shift)


    fig, axs = plt.subplots(1,2,subplot_kw=dict(projection = lev4.wcs_list[0].dropaxis(0)))
    axs[0].set_xlabel('Solar X (arcsec)')
    axs[0].set_ylabel('Solar Y (arcsec)')
    axs[1].set_xlabel('Solar X (arcsec)')
    axs[1].set_ylabel('Solar Y (arcsec)')

    def update(frame,list,list2):
        vel_per_pixel = 18
        axs[0].imshow(list[frame][event[0],event[1]],vmax = np.percentile(np.array(list),99.7))
        axs[1].imshow(list2[frame][event[0],event[1]]*vel_per_pixel,cmap = 'RdBu_r',vmin = -100, vmax = 100)


    ani = FuncAnimation(fig,update,np.arange(len(int)),fargs = (int,shifts))

    save_path = pathlib.Path(__file__).parent / 'figures/' / html_path
    save_path.parent.mkdir(parents = True, exist_ok = True)
    with open(save_path, 'w') as f:
        print(ani.to_html5_video(), file = f)




