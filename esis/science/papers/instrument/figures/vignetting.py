import pathlib
import matplotlib.figure
import matplotlib.pyplot as plt
import pylatex
import kgpy.latex
from .. import optics as optics_factories
from . import formatting
from . import caching

__all__ = [
    'figure_mpl',
    'pdf',
    'figure',
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


def figure() -> pylatex.Figure:
    result = pylatex.Figure()
    result.add_image(str(pdf()), width=None)
    result.add_caption(pylatex.NoEscape(
        r"""
\roy{(Top) 2D histogram counting the number of rays that were unvignetted by the \ESIS\ optical 
system as a function of field position.
The count is normalized to the maximum number of unvignetted rays at any field point.
The field and pupil grids have the same parameters as the grid for Figure~\ref{fig:spotSize}.
(Bottom) Residual between the top histogram and the vignetting model described in Table~\ref{table:vignetting}}"""
    ))
    result.append(kgpy.latex.Label('fig:vignetting'))
    return result
