import esis.data.level_3
import matplotlib.pyplot as plt
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import astropy.units as u
from kgpy.observatories.eis import eis_spectrograph
import kgpy


from esis.data import level_3
from irispy.io import read_iris_sji_level2_fits as read_sji
import irispy.io



if __name__ == '__main__':
    lw = 2

    lev3 = level_3.Level_3.from_pickle(level_3.ov_final_path)
    print(lev3.observation.wcs)

    fig, ax = plt.subplots(subplot_kw=dict(projection=lev3.observation.wcs[15, 1]))
    ax.imshow(lev3.observation.data[15, 1])

    # IRIS SJI
    sji_path = pathlib.Path(__file__).parent / 'data/iris/iris_l2_20190930_171911_3604109624_SJI_1400_t000.fits'
    sji = read_sji(str(sji_path))

    dims = sji.data.shape

    _, ypts, xpts = sji.pixel_to_world(np.zeros(4) * u.pix,
                                       [0, dims[1] - 1, dims[1] - 1, 0] * u.pix,
                                       [0, 0, dims[2] - 1, dims[2] - 1] * u.pix)
    iris_points = np.array([xpts.value, ypts.value, ]).T
    slitjaw = patches.Polygon(iris_points, closed=True, edgecolor='b', facecolor='none', linewidth=lw,
                              transform=ax.get_transform('world'))
    ax.add_patch(slitjaw)

    # IRIS Slit

    launch_raster_dir = pathlib.Path(__file__).parent / 'data/iris/iris_l2_20190930_171911_3604109624_raster'

    sp = irispy.io.read_iris_spectrograph_level2_fits(launch_raster_dir.glob('*.fits'), uncertainty=False)

    si_iv = sp['Si IV 1394'][0]
    # si_iv = sp.data[0]

    dims = si_iv[:, :, 0].data.shape
    xpts, ypts = si_iv[:, :, 0].pixel_to_world((np.array([0, 0, 1, 1, 2, 2, 3, 3]) + 2) * u.pix,
                                               [0, dims[1] - 1, dims[1] - 1, 0, 0, dims[1] - 1, dims[1] - 1,
                                                0] * u.pix, )

    iris_points = np.array([xpts.value, ypts.value, ]).T
    slit = patches.Polygon(iris_points, closed=True, edgecolor='b', facecolor='none', linewidth=lw,
                           transform=ax.get_transform('world'))
    ax.add_patch(slit)

    # EIS

    eis_dir = pathlib.Path(__file__).parent / 'data/eis/pre_launch/l1'
    eis_sp = eis_spectrograph.read_eis_spectrograph_level1_fits(eis_dir.glob('*.fits'), uncertainty=False)

    he_ii = eis_sp.data['He II 256.320']
    dims = he_ii[0][:, :, 0].data.shape
    print(dims)
    xpts, ypts = he_ii[0][:, :, 0].pixel_to_world(
        (np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10])) * u.pix,
        [0, dims[1] - 1, dims[1] - 1, 0,
         0, dims[1] - 1, dims[1] - 1, 0,
         0, dims[1] - 1, dims[1] - 1, 0,
         0, dims[1] - 1, dims[1] - 1, 0,
         0, dims[1] - 1, dims[1] - 1, 0,
         0, dims[1] - 1
         ] * u.pix, )

    eis_points = np.array([xpts.value, ypts.value, ]).T
    eis_slit = patches.Polygon(eis_points / 3600, closed=False, edgecolor='r', facecolor='none', linewidth=lw,
                               transform=ax.get_transform('world'))
    ax.add_patch(eis_slit)

    # print(he_ii.data.shape)
    print(he_ii[0].wcs)
    print(he_ii[0].data.shape)

    fig2, ax2 = plt.subplots(subplot_kw=dict(projection=he_ii[0].wcs[..., 0]))
    ax2.imshow(np.sum(kgpy.rebin(he_ii[0].data, (40, 1, 1)), -1))

    plt.show()