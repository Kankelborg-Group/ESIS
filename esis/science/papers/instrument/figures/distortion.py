import pathlib
import matplotlib.figure
import matplotlib.pyplot as plt
import pylatex
import kgpy.latex
import esis.optics
from .. import optics as optics_factories
from . import formatting
from . import caching

__all__ = [
    'figure_mpl',
    'pdf',
]


def figure_mpl() -> matplotlib.figure.Figure:
    optics = esis.optics.design.final(**optics_factories.default_kwargs)
    optics.num_emission_lines = 1
    fig, ax = plt.subplots(
        figsize=(formatting.column_width, 3.5),
        constrained_layout=True,
    )
    # fig.set_constrained_layout_pads(h_pad=.15)
    optics.plot_field_stop_distortion(
        ax=ax,
        digits_after_decimal=formatting.digits_after_decimal,
        use_latex=True,
    )
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center')
    return fig


def pdf() -> pathlib.Path:
    return caching.cache_pdf(figure_mpl)


def figure() -> pylatex.Figure:
    result = pylatex.Figure()
    result.add_image(str(pdf()), width=None)
    result.add_caption(pylatex.NoEscape(
        r"""
\roy{Plot of the magnified, undistorted field stop aperture vs. the distorted \OV\ image of the 
field stop aperture on the \ESIS\ detector.
The magnification factor used for the undistorted field stop aperture is the ratio of the grating exit arm to the 
grating entrance arm (\armRatio).
The distorted image of the field stop aperture was calculated using the \ESIS\ distortion model, described in 
Table~\ref{table:distortion}.}"""
    ))
    result.append(kgpy.latex.Label('fig:distortion'))
    return result
