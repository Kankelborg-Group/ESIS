import typing as typ
import pathlib
import matplotlib.figure
import matplotlib.axes
import matplotlib.pyplot as plt

__all__ = [
    'cache_pdf'
]


def cache_pdf(fig_factory: typ.Callable[[], matplotlib.figure.Figure]) -> pathlib.Path:
    path = pathlib.Path(__file__).parent / (fig_factory.__name__ + '.pdf')
    if not path.exists():
        fig = fig_factory()
        fig.savefig(
            fname=path,
            # bbox_inches='tight',
            # pad_inches=0.04,
            # facecolor='lightblue'
        )
        plt.close(fig)
    return path
