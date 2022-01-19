import pylatex
import kgpy.latex

__all__ = [
    'figure',
]


def figure() -> pylatex.Figure:
    result = pylatex.Figure()
    result.add_image('figures/old/ESIS_Cameras_1gr_text', width=kgpy.latex.columnwidth)
    result.add_caption(pylatex.NoEscape(
        r"""
\ESIS\ camera assembly as built by \MSFCShort.  
Thin film filters and filter tubes are not installed in this image."""
    ))
    result.append(kgpy.latex.Label('F-cameras'))
    return result
