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
    optics = optics_factories.as_designed_single_channel()
    fig, axs = plt.subplots(
        ncols=optics_factories.num_emission_lines_default,
        figsize=(formatting.text_width, 2.5),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )
    optics.rays_output.plot_spot_size_vs_field(
        axs=axs,
        digits_after_decimal=formatting.digits_after_decimal,
    )
    return fig


def pdf() -> pathlib.Path:
    return caching.cache_pdf(figure_mpl)