import numpy as np
import matplotlib.pyplot as plt
from kgpy.observatories.sdo import hmi, aia
from esis.data import level_3
import astropy.units as u
from kgpy.plot import CubeSlicer
import pathlib

if __name__ == '__main__':

    lev3 = level_3.Level_3.from_pickle(level_3.ov_final_path)
    times = lev3.time

    aia_obj = aia.AIA.from_time_range(times[0]-700*u.s, times[-1]+700 * u.s, user_email='jacobdparker@gmail.com',
                                  channels=[94*u.AA])
    slice = CubeSlicer(aia_obj.intensity[:, 0, ...].value, projection=aia_obj.wcs[0,0], origin='lower',
                       vmax=np.percentile(aia_obj.intensity[:, 0, ...].value, 99.9))
    plt.show()


    # hmi_download_path = pathlib.Path(__file__).parent / 'hmi_data'
    # # if hmi_download_path.is_dir():
    # #     l1_path = hmi_download_path / 'level_1'
    # #     hmi_path_array = np.array(sorted(list(l1_path.glob('*'))))
    # #     hmi_obj = hmi.HMI.from_path_array(hmi_path_array)
    # # else:
    # hmi_obj = hmi.HMI.from_time_range(times[0] - 700 * u.s, times[-1] + 700 * u.s, user_email='jacobdparker@gmail.com',
    #                                       download_path=hmi_download_path)
    # scale = 30
    # slice = CubeSlicer(hmi_obj.intensity[:, 0, ...], vmin=-scale, vmax=scale, projection=hmi_obj.wcs[0,0], origin='lower')
    # plt.show()

    #
    # fig = plt.figure()
    # ax1 = fig.add_subplot(2,2,1,projection=aia.wcs[0,0])
    # ax1.set_xlabel('Solar X (arcsec)')
    # ax1.set_ylabel('Solar Y (arcsec)')
    #
    # ax2 = fig.add_subplot(2,2,2)
    # ax2.set_xlabel('Solar X (arcsec)')
    # ax2.set_ylabel('Solar Y (arcsec)')
    #
    # ax3 = fig.add_subplot(2,2,3)
    # ax3.set_xlabel('Solar X (arcsec)')
    # ax3.set_ylabel('Solar Y (arcsec)')
    #
    # ax4 = fig.add_subplot(2,2,4)
    # ax4.set_xlabel('Solar X (arcsec)')
    # ax4.set_ylabel('Solar Y (arcsec)')
    #
    # p=99.9
    # im1 = ax1.imshow(aia[0,0], vmax=np.percentile(aia[:,0,...], p), cmap='sdoaia304')
    #
    #
    # def update(seq):
    #     im1.set_data(aia[seq,0])
    #
    #     # im2.set_data(shifts[seq]*18)
    #     # im3.set_data(l4_int[seq])
    #     # im4.set_data(aia_304_imgs[seq])
    #     # im5.set_data(widths[seq]*18)
    #     # line1[0].set_ydata(l4.cube_list[seq][pix1])
    #     # line2[0].set_ydata(l4.cube_list[seq][pix2])
    #     # line3[0].set_ydata(l4.cube_list[seq][pix3])
    #
    #
    # ani = FuncAnimation(fig, update, frames=10)
    # plt.show()
    #
    # # save_path = pathlib.Path(__file__).parent / 'figures/' / 'main_event_movie.avi'
    # # save_path.parent.mkdir(parents=True, exist_ok=True)
    # # ani.save(save_path,'ffmpeg',dpi = 200)
