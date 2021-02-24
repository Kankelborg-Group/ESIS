from esis.data import level_1
from kgpy.plot import CubeSlicer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pathlib
import numpy as np

if __name__ == '__main__':
    camera = 1

    lev1 = level_1.Level_1.from_pickle()
    lev1.intensity = np.flip(lev1.intensity, axis=-2)

    fig , ax = plt.subplots(figsize = (12,5), subplot_kw= dict(title = 'ESIS Level 1 Movie (Camera 2)'))
    ax.set_xlabel('Detector Pixels')
    ax.set_ylabel('Detector Pixels')
    img = ax.imshow(lev1.intensity[0,camera].value,origin = 'lower',
                    vmax = np.percentile(lev1.intensity[:,camera,...].value, 99.99))
    fig.colorbar(img, ax = ax, label = 'Intensity (630 Angstrom Photons)')


    t = ax.annotate(lev1.time[0,camera].to_value('isot'),(5,5), color = 'w')
    y_height = 960
    he_i = ax.annotate('He I', (250, y_height), color = 'w')
    mg_x = ax.annotate('Mg X', (950, y_height), color = 'w')
    o_v = ax.annotate('O V', (1500, y_height), color = 'w')

    def update(frame):
        img.set_data(lev1.intensity[frame,camera].value)
        t.set_text(lev1.time[frame,camera].to_value('isot'))


    ani = FuncAnimation(fig, update, np.arange(lev1.intensity.shape[0]))

    plt.show()

    save_path = pathlib.Path(__file__).parent / 'figures/' / 'l1_movie.avi'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(save_path,'ffmpeg',dpi = 200)
