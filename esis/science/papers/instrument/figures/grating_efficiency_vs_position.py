import pathlib
import matplotlib.figure
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import pylatex
import kgpy.latex
import esis.optics
from . import formatting
from . import caching

__all__ = [
    'figure_mpl',
    'pdf',
    'figure',
]


def figure_mpl() -> matplotlib.figure.Figure:
    with astropy.visualization.quantity_support():
        fig, axs = plt.subplots(
            ncols=2,
            sharey=True,
            figsize=(formatting.text_width, 2.5),
            constrained_layout=True,
        )

        position_x, position_y, wavelength, efficiency = esis.optics.grating.efficiency.vs_position_x()
        axs[0].plot(
            position_x,
            efficiency.to(u.percent),
            label='grating 017',
        )
        axs[0].set_xlabel(f'$x$ position ({axs[0].get_xlabel()})')
        axs[0].set_ylabel(f'efficiency ({axs[0].get_ylabel()})')

        position_x, position_y, wavelength, efficiency = esis.optics.grating.efficiency.vs_position_y()
        axs[1].plot(
            position_y,
            efficiency,
            label='grating 017'
        )
        axs[1].set_xlabel(f'$y$ position ({axs[1].get_xlabel()})')
        axs[1].set_ylabel(None)

        return fig


def pdf() -> pathlib.Path:
    return caching.cache_pdf(figure_mpl)


def figure() -> pylatex.Figure:
    result = pylatex.Figure()
    result._star_latex_name = True
    result.add_image(str(pdf()), width=None)
    result.add_caption(pylatex.NoEscape(
        r"""
\roy{Channel \testGratingChannelIndex\ grating efficiency at \gratingTestWavelength\ vs. position for two orthogonal slices across the optical 
surface on \testGratingDate.}"""
    ))
    result.append(kgpy.latex.Label('fig:gratingEfficiencyVsPosition'))
    return result
