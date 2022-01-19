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


def figure() -> pylatex.Figure:
    result = pylatex.Figure()
    result._star_latex_name = True
    result.add_image(str(pdf()), width=None)
    result.add_caption(pylatex.NoEscape(
        r"""
\roy{Magnitude of the residual between a linear distortion model and the raytrace model (top) and between a quadratic 
distortion model and the raytrace model (bottom). This figure demonstrates that a quadratic distortion model is
sufficient to achieve sub-pixel accuracy.}"""
    ))
    result.append(kgpy.latex.Label('fig:distortionResidual'))
    return result
