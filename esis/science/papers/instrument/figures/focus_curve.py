import pathlib
import matplotlib.figure
import matplotlib.pyplot as plt
import astropy.units as u
import esis.optics
import pylatex
import kgpy.latex
from .. import optics as optics_factories
from . import formatting
from . import caching

__all__ = [
    'figure_mpl',
    'pdf',
]


def figure_mpl() -> matplotlib.figure.Figure:
    optics = esis.optics.design.final(
        pupil_samples=11,
        field_samples=1,
        # field_is_stratified_random=True,
        all_channels=False,
    )
    optics.num_emission_lines = optics_factories.num_emission_lines_default
    fig, ax = plt.subplots(
        figsize=(formatting.column_width, 2.5),
        constrained_layout=True,
    )

    optics.plot_focus_curve(
        ax=ax,
        delta_detector=5 * u.mm,
        num_samples=51,
        digits_after_decimal=2,
        use_latex=True,
    )
    ax.legend(bbox_to_anchor=(0.5, -0.3), loc='upper center', ncol=2)
    fig.set_constrained_layout_pads(w_pad=.10, h_pad=0.10)

    return fig


def pdf() -> pathlib.Path:
    return caching.cache_pdf(figure_mpl)


def figure() -> pylatex.Figure:
    result = pylatex.Figure()
    result.add_image(str(pdf()), width=None)
    result.add_caption(pylatex.NoEscape(
        r"""
\roy{Focus curve for the field angle at the middle of the \ESIS\ \FOV\ for the 
\defaultNumEmissionLines\ brightest wavelengths in the passband.
}"""
    ))
    result.append(kgpy.latex.Label('fig:focusCurve'))
    return result
