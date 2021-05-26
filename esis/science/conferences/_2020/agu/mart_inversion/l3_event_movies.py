import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from esis.data import level_3, level_4
import pathlib
from esis.flight import l3_events

if __name__ == '__main__':
    l3 = level_3.Level_3.from_pickle(level_3.ov_final_path)

    event = l3_events.perfectx

    l4 = level_4.Level_4.from_pickle(event.mart_inverted_pickle_path)
    l4_int = np.array(l4.integrated_intensity)
    bright_pix = np.unravel_index(l4_int.argmax(),l4_int.shape)
    peak_profile = np.array(l4.cube_list)[:,bright_pix[1],bright_pix[2],:].max()

    l3_event = l3.observation[:, :, event.location[0], event.location[1]]
    times = l3.time
    l3_dif = l3_event.data[:,1,...] - l3_event.data[:,2,...]

    projection = l3_event.wcs.dropaxis(-1).dropaxis(-1)

    fig = plt.figure(figsize = (15,4))
    ax1 = plt.subplot(1, 3, 1, projection = projection, title = 'Camera 2 - Camera 3')
    ax1.set_xlabel('Solar X')
    ax1.set_ylabel('Solar Y')
    t = ax1.annotate(times[0],(3,3),color = 'w')

    ax2 = plt.subplot(1, 3, 2, projection = projection, title = 'Summed Intensity (Inverted Data)')
    ax2.set_xlabel('Solar X')
    ax2.set_ylabel('Solar Y')

    ax3 = plt.subplot(1, 3, 3, projection = l4.wcs_list[0].dropaxis(-1).dropaxis(-1), title = 'Line Profile at Red Pixel')
    ax3.set_ylabel('Intensity (630 Angstrom Photons)')

    scale = np.percentile(l3_event.data[:,1,...],95)
    img1 = ax1.imshow(l3_dif[0], vmax=scale, vmin=-scale)

    vmax = np.percentile(l4_int, 99.9)
    img2 = ax2.imshow(l4_int[0],vmax = vmax)
    profile = ax3.plot(l4.cube_list[0][bright_pix[1],bright_pix[2],:])
    ax3.set_ylim([0,peak_profile])
    dot = ax2.plot(bright_pix[2],bright_pix[1], marker = '.', color = 'r')
    # vline = ax2.axvline(brightest_pix[2])
    # hline = ax2.axhline(brightest_pix[1])



    def update(frame):
        img1.set_data(l3_dif[frame])
        img2.set_data(l4_int[frame])
        profile[0].set_ydata(l4.cube_list[frame][bright_pix[1],bright_pix[2],:])
        t.set_text(times[frame])

    ani = FuncAnimation(fig, update, np.arange(l3.observation.data.shape[0]))
    plt.show()

    save_path = pathlib.Path(__file__).parent / 'figures/' / (event.name +'_movie.avi')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(save_path,'ffmpeg',dpi = 200)


