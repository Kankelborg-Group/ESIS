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
    optics = optics_factories.as_measured_single_channel()
    fig, axs = plt.subplots(
        figsize=(formatting.column_width, 2.9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    model = optics.system.rays_output_resample_entrance.vignetting
    model.plot_unvignetted(axs=axs[0], wavelength_name=optics.bunch.fullname(formatting.digits_after_decimal))
    return fig


def pdf() -> pathlib.Path:
    return caching.cache_pdf(figure_mpl)
