import pylatex
from . import moses_limitations
from . import esis_features

__all__ = [
    'section'
]


def section() -> pylatex.Section:
    result = pylatex.Section(pylatex.NoEscape('The \ESIS\ Concept'))
    result.escape = False
    result.append(
        r"""
A primary goal of the \ESIS\ instrument is to improve upon the imaging spectroscopy demonstrated by \MOSES.  
Therefore, the design of the new instrument draws heavily from experiences and lessons learned through two flights of the \MOSES\ instrument.
\ESIS\ and \MOSES\ are both \CTIS\ instruments.
As such, both produce \sout{dispersed images} \roy{overlappograms} of a narrow portion of the solar spectrum, with the goal of enabling the reconstruction of a spectral line profile at every point in the field of view.
The similarities end there, however, as the optical layout of \ESIS\ differs significantly from that of \MOSES.
In this section, we detail some difficulties and limitations encountered with \MOSES, then describe how the new design of \ESIS\ addresses these issues."""
    )
    result.append(moses_limitations.subsection())
    result.append(esis_features.subsection())
    return result
