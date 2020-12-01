"""
Package is for code and information relating to the 2019 flight
"""
__all__ = [
    'optics',
    'raw_img_dir',
    'level_0',
    'num_dark_safety_frames'
]

import pathlib
from .. import data
from . import optics

raw_img_dir = pathlib.Path(__file__).parent / 'images'
num_dark_safety_frames = 3


def level_0(caching: bool = False) -> data.Level_0:
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
    return data.Level_0.from_directory(
        directory=raw_img_dir,
        detector=optics.as_measured().detector,
        caching=caching,
        num_dark_safety_frames=num_dark_safety_frames,
    )
