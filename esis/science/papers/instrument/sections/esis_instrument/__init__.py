import pylatex
import kgpy.latex
from ... import tables
from . import optics
from . import optimization_and_tolerancing
from . import vignetting
from . import distortion
from . import coatings_and_filters
from . import sensitivity_and_cadence
from . import alignment_and_focus
from . import apertures_and_baffles
from . import cameras
from . import avionics
from . import pointing_system
from . import mechanical

__all__ = [
    'optics',
    'optimization_and_tolerancing',
    'vignetting',
    'distortion',
    'coatings_and_filters',
    'sensitivity_and_cadence',
    'alignment_and_focus',
    'apertures_and_baffles',
    'cameras',
    'avionics',
    'pointing_system',
    'mechanical',
]


def section(doc: kgpy.latex.Document) -> pylatex.Section:
    result = pylatex.Section(pylatex.NoEscape(r'The \ESIS\ Instrument'))
    result.escape = False
    result.append(pylatex.NoEscape(
        r"""\ESIS\ is a multi-projection slitless spectrograph that obtains line intensities, Doppler shifts, and 
widths in a single snapshot over a 2D \FOV.
Starting from the notional instrument described in Sec.~\ref{sec:TheESISConcept}, \ESIS\ has been designed to ensure all 
of the science requirements set forth in Table~\ref{table:scireq} are met.
The final design parameters are summarized in Table~\ref{table:prescription}.

A schematic diagram of a single \ESIS\ channel is presented in Fig.~\ref{fig:schematic}a, while the mechanical features 
of the primary mirror and gratings are detailed in Figs.~\ref{fig:schematic}b and \ref{fig:schematic}c, respectively."""
    ))
    result.append(tables.prescription.table())
    result.append(optics.subsection())
    result.append(optimization_and_tolerancing.subsection(doc))
    result.append(vignetting.subsection())
    result.append(distortion.subsection())
    result.append(coatings_and_filters.subsection())
    result.append(sensitivity_and_cadence.subsection(doc))
    result.append(alignment_and_focus.subsection())
    result.append(apertures_and_baffles.subsection())
    result.append(cameras.subsection())
    result.append(avionics.subsection())
    result.append(pointing_system.subsection())
    result.append(mechanical.subsection())
    return result
