import pylatex
from ... import figures
from ... import tables

__all__ = [
    'subsection',
]


def subsection() -> pylatex.Subsection:
    result = pylatex.Subsection('Science Requirements')
    result.escape = False
    result.append(
        r"""
\ESIS\ will investigate two science targets; 
reconnection in explosive events, and the transport of mass and energy through the transition region.
The latter may take many forms, from \MHD\ waves of various modes to \EUV\ jets or macro-spicules.
To fulfill these goals, \ESIS\ will obtain simultaneous intensity, Doppler shift and line width images of the \OV\ line 
in the solar transition region at rapid cadence.
This is a lower \TR\ line (\SI{.25}{\mega\kelvin}).
The bright, optically thin \OVion\ emission line is well isolated except for the two coronal \MgXion\ lines.
These coronal lines can be viewed as contamination or as a bonus;
we expect that with the \numChannelsWords\ \ESIS\ projections it will be possible to separate the \OVion\ emission from 
that of \MgXion.
From the important temporal, spatial, and velocity scales referenced Sections~\ref{subsec:MagneticReconnectionEvents} 
and \ref{subsec:EnergyTransfer} we define the instrument requirements in Table~\ref{table:scireq} that are needed to 
meet our science goals."""
    )
    result.append(figures.bunch.figure())
    result.append(tables.requirements.table())
    return result
