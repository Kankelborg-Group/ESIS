import aastex

__all__ = ['figure']


def figure() -> aastex.Figure:

    result = aastex.Figure("fig:mosesSchematic", position='!ht')
    result.add_image('figures/old/MOSES_Schematic', width=aastex.columnwidth)
    result.add_caption(aastex.NoEscape(
        r"""
Schematic diagram of the \MOSES\ instrument.
Incident light on the right forms an undispersed image on the central $m=0$ \CCD.
Dispersed images are formed on the outboard $m=\pm1$ \CCDs."""
    ))
    return result
