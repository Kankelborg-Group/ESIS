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


def figure() -> pylatex.Figure:
    result = pylatex.Figure()
    result._star_latex_name = True
    result.add_image(str(pdf()), width=None)
    result.add_caption(pylatex.NoEscape(
        r"""
\roy{2D histogram of RMS spot sizes for the \defaultNumEmissionLines\ brightest wavelengths in the \ESIS\ passband. 
Each wavelength has $\defaultFieldSamples \times \defaultFieldSamples$ field points across the \FOV, and each field point
has a stratified random grid containing $\defaultPupilSamples \times \defaultPupilSamples$ pupil positions.
The \HeI\ line appears cropped since it is cut off by the edge of the detector.
The images appear flipped compared to Figure~\ref{fig:projections} since the optical system inverts the image of the skyplane.
}"""
    ))
    result.append(kgpy.latex.Label('fig:spotSize'))
    return result
