import pylatex
import kgpy.latex

__all__ = [
    'figure'
]


def figure() -> pylatex.Figure:
    result = pylatex.Figure()
    result._star_latex_name = True
    result.add_image('figures/old/Baffles_1clr', width=kgpy.latex.textwidth)
    result.add_caption(pylatex.NoEscape(
        r"""Model view of \ESIS\ baffle placement and cutouts."""
    ))
    result.append(kgpy.latex.Label('F-Baff1'))
    return result
