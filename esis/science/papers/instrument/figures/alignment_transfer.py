import pylatex
import kgpy.latex

__all__ = [
    'figure',
]


def figure() -> pylatex.Figure:
    result = pylatex.Figure()
    result.add_image('figures/old/Alignment_transfer_1gr_text', width=kgpy.latex.columnwidth)
    result.add_caption(pylatex.NoEscape(
        r"""
\jake{Will update this figure.  Will include a rendering of the secondary mount pointing to the tuffet, grating
backplate, bipod, etc. Capturing the same as before but without TEA.}
\ESIS\ alignment transfer device, consisting of three miniature confocal microscopes that translate along the optical 
axis.  
Trapezoidal grating, bipods, and mounting plate are installed on the tuffet in front of the apparatus 
(left of center)"""
    ))
    result.append(kgpy.latex.Label('F-alt'))
    return result
