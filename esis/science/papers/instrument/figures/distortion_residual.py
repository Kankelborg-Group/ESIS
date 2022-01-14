import pathlib
import matplotlib.figure
import matplotlib.pyplot as plt
from .. import optics as optics_factories
from . import formatting
from . import caching

__all__ = [
    'figure_mpl',
    'pdf',
]


def figure_mpl() -> matplotlib.figure.Figure:
    optics_quadratic = optics_factories.as_designed_single_channel()
    optics_linear = optics_quadratic.copy()
    optics_linear.distortion_polynomial_degree = 1
    optics_linear.update()
    fig, axs = plt.subplots(
        nrows=2,
        ncols=optics_factories.num_emission_lines_default,
        sharex=True,
        sharey=True,
        figsize=(formatting.text_width, 4.4),
        constrained_layout=True,
    )
    distortion_linear = optics_linear.rays_output.distortion
    distortion_quadratic = optics_quadratic.rays_output.distortion

    distortion_linear.plot_residual(
        axs=axs[0],
        use_xlabels=False,
        wavelength_name=optics_linear.bunch.fullname(formatting.digits_after_decimal),
    )
    distortion_quadratic.plot_residual(
        axs=axs[1],
        use_titles=False,
    )

    return fig


def pdf() -> pathlib.Path:
    return caching.cache_pdf(figure_mpl)
