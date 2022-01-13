import pathlib
import matplotlib.figure
import matplotlib.pyplot as plt
import esis.optics
from . import formatting
from . import caching

__all__ = [
    'figure_mpl',
    'pdf',
]


def figure_mpl() -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(formatting.column_width, 3), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=.15)
    # ax.margins(x=.01, y=.01)
    optics = esis.optics.design.final(all_channels=False)
    optics.plot_field_stop_projections_local(
        ax=ax,
        wavelength_color=[
            'darkviolet',
            'indigo',
            'blue',
            'dodgerblue',
            'cyan',
            'green',
            'chartreuse',
            'orange',
            'orangered',
            'red',
        ],
        digits_after_decimal=formatting.digits_after_decimal,
        use_latex=True,
    )
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(0.5, -0.25), loc='upper center', ncol=2)
    return fig


def pdf() -> pathlib.Path:
    return caching.cache_pdf(figure_mpl)
