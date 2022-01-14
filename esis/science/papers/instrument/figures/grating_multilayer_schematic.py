import pathlib
import matplotlib.figure
import matplotlib.pyplot as plt
import esis.optics
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
    optics = esis.optics.design.final(**optics_factories.default_kwargs)
    fig, ax = plt.subplots(
        figsize=(formatting.column_width, 2.2),
        constrained_layout=True,
    )
    optics.grating.surface.material.plot_layers(
        ax=ax,
        layer_material_color=dict(
            Al='lightblue',
            Mg='pink',
            SiC='lightgray',
        ),
        layer_label_x=dict(
            Al=1.0,
            Mg=0.5,
            SiC=0.5,
        ),
        layer_label_x_text=dict(
            Al=1.2,
            Mg=0.5,
            SiC=0.5,
        )
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    fig.set_constrained_layout_pads(w_pad=0.2)
    return fig


def pdf() -> pathlib.Path:
    return caching.cache_pdf(figure_mpl)


def figure() -> pylatex.Figure:
    result = pylatex.Figure()
    result.add_image(str(pdf()), width=None)
    result.add_caption(pylatex.NoEscape(
        r"""
Schematic of the Al/SiC/Mg \roy{\gratingCoatingMaterialShort} multilayer with $N=4$ \roy{$N=\gratingCoatingNumLayers$} layers."""
    ))
    result.append(kgpy.latex.Label('fig:gratingMultilayerSchematic'))
    return result
