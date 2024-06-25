import matplotlib.figure
import matplotlib.pyplot as plt
import pathlib
import aastex
from esis.science.papers.instrument.figures import formatting
from esis.science.papers.instrument.figures import caching

__all__ = [
    'figure',
]

def figure_mpl() -> matplotlib.figure.Figure:

    img_path = pathlib.Path(__file__).parent / 'static/esis_secondary_mount.png'
    fig, ax = plt.subplots(figsize=(formatting.column_width, 3), constrained_layout=True)


    ax.imshow(plt.imread(img_path))
    # ax.margins(x=.01, y=.01)
    #
    # # ax.set_aspect('equal')
    ax.set_axis_off()

    apkw = dict(
        arrowstyle='->',
        linewidth=0.75,
        relpos=(0.5, 0.5),
    )
    kwargs_annotate = dict(
        ha='center',
    )
    ax.annotate(
        text='Tuffet',
        xy=(1000,550),
        xytext=(250,250),
        arrowprops=dict(
            **apkw,
        ),
        # ha='center',
        **kwargs_annotate
    )
    ax.annotate(
        text='Backplate',
        xy=(1010,760),
        xytext=(150,770),
        arrowprops=dict(
            **apkw,
        ),
        # ha='center',
        **kwargs_annotate
    )
    ax.annotate(
        text='Bipod',
        xy=(1130,550),
        xytext=(1460,120),
        arrowprops=dict(
            **apkw,
        ),
        # ha='center',
        **kwargs_annotate
    )
    ax.annotate(
        text='Grating',
        xy=(1340,650),
        xytext=(1740,700),
        arrowprops=dict(
            **apkw,
        ),
        # ha='center',
        **kwargs_annotate
    )
    #
    #
    # ax.set_ylabel(None)
    # ax.set_yticks([])
    # ax.set_xticks([])
    # ax.set_xlim(right=250)
    # colorbar.remove()

    return fig


def pdf() -> pathlib.Path:
    return caching.cache_pdf(figure_mpl)

def figure() -> aastex.Figure:
    result = aastex.Figure("F-alt")
    result.add_image(pdf(), width=formatting.column_width)
    result.add_caption(aastex.NoEscape(
        r""" A solid model of the \ESIS\ secondary mount.
            Each trapezoidal grating is supported and attached to it's backplate by three bonded, titanium bipods.
            All mounted gratings sit on the "tuffet" and are mounted with three spherical washers.
            This allows independent tip/tilt adjustment of each grating, and for all of them to be removed/reinstalled
            with out compromising alignment and focus.
             """
    ))
    return result

if __name__ == '__main__':
    figure = figure_mpl()
    plt.show()