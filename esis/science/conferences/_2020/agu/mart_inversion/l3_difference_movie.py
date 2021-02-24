from esis.data import level_3
from kgpy.plot import CubeSlicer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pathlib
import numpy as np

if __name__ == '__main__':

    lev3 = level_3.Level_3.from_pickle(level_3.ov_final_path)
    lev3_dif = lev3.observation.data[:,1,...]-lev3.observation.data[:,2,...]
    times = lev3.time

    slicer = CubeSlicer(lev3_dif,origin = 'lower')
    plt.show()

    scale = 50
    fig, ax = plt.subplots(subplot_kw=dict(projection = lev3.observation.wcs.dropaxis(-1).dropaxis(-1),
                                           title = 'Level 3 Difference (Camera 2 Minus Camera 3)'))
    ax.set_xlabel('Solar X (arcsec)')
    ax.set_ylabel('Solar Y (arcsec)')
    img = ax.imshow(lev3_dif[0,...], vmin = -scale, vmax = scale)
    t = ax.annotate(str(times[0]),(5,5))
    fig.colorbar(img,ax = ax,label = "Intensity Difference (photons)")


    def update(frame, lev3_dif, times):

        img.set_data(lev3_dif[frame,...])
        t.set_text(str(times[frame]))



    ani = FuncAnimation(fig, update, np.arange(lev3.observation.data.shape[0]), fargs=(lev3_dif,times))

    # save_path = pathlib.Path(__file__).parent / 'figures/' / 'dif_movie.html'
    # save_path.parent.mkdir(parents=True, exist_ok=True)
    # with open(save_path, 'w') as f:
    #     print(ani.to_html5_video(), file=f)


    plt.show()

    save_path = pathlib.Path(__file__).parent / 'figures/' / 'dif_movie.avi'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(save_path,'ffmpeg',dpi = 200)
