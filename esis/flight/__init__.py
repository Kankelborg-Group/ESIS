"""
Package is for code and information relating to the 2019 flight
"""
__all__ = [
    'optics',
    'raw_img_dir',
    'level_0',
    'level_1_cache',
    'level_1'
]

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from kgpy import plot
from kgpy.observatories import iris
from .. import data
from . import optics, nsroc


raw_img_dir = pathlib.Path(__file__).parent / 'images'

level_1_cache = pathlib.Path(__file__).parent / 'level_1.pickle'
cnn_inversion_cache = pathlib.Path(__file__).parent / 'cnn_inversion.pickle'
level_2_cache = pathlib.Path(__file__).parent / 'level_2.pickle'
level_3_cache = data.level_3.ov_final_path


def level_0() -> data.Level_0:
    """
    Compute a :class:`esis.data.level_0.Level_0` instance for the 2019 flight.

    ##############################################
    myModule1 Title
    ##############################################



    The Level-0 dataset represents the raw data obtained from the MSFC cameras.

    In the example below, we've plotted an animation of the raw data obtained during the 2019 ESIS flight.

    **********************************************
    First Section title
    **********************************************

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        from IPython.display import HTML
        import esis


        %time level_0 = esis.flight.level_0(caching=True)

        fig_intensity_ani, ax_intensity_ani = plt.subplots(
            ncols=4, figsize=(9.5, 7), sharex=True, sharey=True, constrained_layout=True, squeeze=False)
        %time intensity_ani = level_0.animate_intensity(ax_intensity_ani)
        %time intensity_ani_html = intensity_ani.to_jshtml()
        plt.close(fig_intensity_ani)
        HTML(intensity_ani_html)



    The raw FPGA measurements are also included in the Level-0 dataset.
    Plotted below are the FPGA measurements from the FITS header during the 2019 ESIS flight.

    .. jupyter-execute::

        _, axs_fpga = plt.subplots(nrows=5, figsize=(9.5, 7), sharex=True, constrained_layout=True, dpi=200)
        _ = level_0.plot_fpga_stats_vs_index(axs_fpga)


    Here, we've plotted the mean value of the intensity vs. sequence index.
    We can see that the background of all four channels is approximately 3600 adu and the EUV signal from then sun is
    about 50 adu above the background.

    .. jupyter-execute::

        _, axs_bias_sub = plt.subplots(nrows=3, figsize=(9.5, 6), sharex=True, constrained_layout=True, dpi=200)
        %time _ = level_0.plot_bias_subtraction_vs_index(axs_bias_sub)

    .. jupyter-execute:

        fig_dark, axs_dark = plt.subplots(ncols=4, figsize=(9.5, 6), sharex=True, sharey=True, constrained_layout=True)
        _ = level_0.plot_dark(axs=axs_dark)

    Parameters
    ----------
    caching

    Returns
    -------
    esis.data.level_0.Level_0





    """
    trajectory = nsroc.trajectory()
    return data.Level_0.from_directory(
        directory=raw_img_dir,
        optics=optics.as_measured(),
        trajectory=trajectory,
        timeline=nsroc.timeline(),
    )


def level_1() -> data.Level_1:

    if level_1_cache.exists():
        return data.Level_1.from_pickle(level_1_cache)

    else:
        l1 = data.Level_1.from_level_0(level_0())
        l1.to_pickle(level_1_cache)
        return l1


def cnn_inversion() -> data.inversion.cnn.CNN:
    if cnn_inversion_cache.exists():
        return data.inversion.cnn.CNN.from_pickle(cnn_inversion_cache)

    else:
        cube = iris.mosaics.load_index()

        spatial_rebin_factor = 2
        spectral_rebin_factor = 2
        spectral_trim = cube.num_wavelength // spectral_rebin_factor * spectral_rebin_factor

        rsh = cube.shape[:~2]
        rsh += (cube.shape[~2] // spatial_rebin_factor, spatial_rebin_factor, )
        rsh += (cube.shape[~1] // spatial_rebin_factor, spatial_rebin_factor, )
        rsh += (cube.shape[~0] // spectral_rebin_factor, spectral_rebin_factor, )

        cube.intensity = cube.intensity[..., :spectral_trim]
        cube.intensity = cube.intensity.reshape(rsh)
        cube.intensity = cube.intensity.sum((~4, ~2, ~0))

        cube.intensity_uncertainty = cube.intensity_uncertainty[..., :spectral_trim]
        cube.intensity_uncertainty = cube.intensity_uncertainty.reshape(rsh)
        cube.intensity_uncertainty = cube.intensity_uncertainty.sum((~4, ~2, ~0))

        for w, _ in enumerate(cube.wcs.flat):
            cube.wcs.flat[w].wcs.cdelt[~1:] /= 3
            cube.wcs.flat[w].wcs.cdelt[~1:] *= spatial_rebin_factor
            cube.wcs.flat[w].wcs.crpix[~1:] /= spatial_rebin_factor
            cube.wcs.flat[w].wcs.cdelt[~2] *= spectral_rebin_factor
            cube.wcs.flat[w].wcs.crpix[~2] /= spectral_rebin_factor
            cube.wcs.flat[w].wcs.crval[~1] = 50
            cube.wcs.flat[w].wcs.crval[~0] = -100

        np.random.seed(42)
        random_ind = np.random.permutation(cube.num_times)
        training_ind, validation_ind = random_ind.reshape((2, -1))

        cube_training = iris.mosaics.Cube(
            intensity=cube.intensity[training_ind],
            intensity_uncertainty=cube.intensity_uncertainty[training_ind],
            wcs=cube.wcs[training_ind],
            time=cube.time[training_ind],
            time_index=cube.time_index[training_ind],
            channel=cube.channel.copy(),
            exposure_length=cube.exposure_length[training_ind],
        )

        cube_validation = iris.mosaics.Cube(
            intensity=cube.intensity[validation_ind],
            intensity_uncertainty=cube.intensity_uncertainty[validation_ind],
            wcs=cube.wcs[validation_ind],
            time=cube.time[validation_ind],
            time_index=cube.time_index[validation_ind],
            channel=cube.channel.copy(),
            exposure_length=cube.exposure_length[validation_ind],
        )

        cnn = data.inversion.cnn.CNN.train(
            model_forward=optics.as_flown(disk_cache=optics.as_flown_cache.parent / 'optics_as_flown_1.pickle'),
            cube_training=cube_training,
            cube_validation=cube_validation,
        )

        cnn.to_pickle(cnn_inversion_cache)

        return cnn


def level_2() -> data.Level_2:
    if level_2_cache.exists():
        return data.Level_2.from_pickle()

    else:
        pass

def level_3(despike=True, full_prep=False) -> data.Level_3:

    if level_3_cache.exists():
        return data.Level_3.from_pickle(level_3_cache)

    else:
        l3 = data.level_3.full_level3_prep(despike=despike, full_prep=full_prep)
        return l3
