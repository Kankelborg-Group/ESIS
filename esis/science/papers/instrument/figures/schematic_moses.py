import pylatex
import kgpy.latex

__all__ = ['figure']


def figure() -> pylatex.Figure:

    result = pylatex.Figure(position='!ht')
    result.escape = False
    result.add_image('figures/old/MOSES_Schematic', width=kgpy.latex.columnwidth)
    result.add_caption(pylatex.NoEscape(
        r"""
Schematic diagram of the \MOSES\ instrument.
Incident light on the right forms an undispersed image on the central $m=0$ \CCD.
Dispersed images are formed on the outboard $m=\pm1$ \CCDs."""
    ))
    result.append(kgpy.latex.Label('fig:mosesSchematic'))
    return result
