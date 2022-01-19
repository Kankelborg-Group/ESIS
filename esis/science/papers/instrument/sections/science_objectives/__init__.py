import pylatex
from . import magnetic_reconnection_events
from . import energy_transfer
from . import requirements

__all__ = [
    'magnetic_reconnection_events',
    'energy_transfer',
    'requirements',
    'section',
]


def section() -> pylatex.Section:
    result = pylatex.Section('Science Objectives')
    result.escape = False
    result.append(
        r"""
The previous section discussed the qualitative design aspects of \ESIS\ learned from experience with the 
\MOSES\ instrument.  
\MOSES, in turn, demonstrated a working concept of simultaneous \EUV\ imaging and spectroscopy.
This concept adds a unique capability to the science that we can obtain from the \EUV\ solar atmosphere.
\ESIS, sharing the same payload volume as \MOSES, is manifested to fly in 2019.
In this section, we set forth specific scientific objectives for the combined \ESIS/\MOSES\ mission.
From these objectives, and with an eye toward synergistic operation of \MOSES\ and \ESIS, in 
\S\,\ref{subsec:ScienceRequirements} we derive the quantitative science requirements that drive the \ESIS\ design.

\jake{Early flights of \MOSES\ demonstrated a working concept of simultaneous \EUV\ imaging and spectroscopy. 
This concept adds a unique capability to the science that we can obtain from the \EUV\ solar atmosphere. 
\ESIS\ as designed improves upon the \MOSES\ concept, as discussed in the previous section, and therefore improves our ability 
to accomplish our scientific objectives. 
In this section, we set forth the specific scientific objectives of the \ESIS mission. 
It is from these objectives that we derived the quantitative science requirements (\S\,\ref{subsec:ScienceRequirements}) 
that drove the \ESIS\ design. }

The \ESIS\ mission was designed to achieve the following two overarching science goals: \begin{inparaenum}[(1)] 
\item observe magnetic reconnection in the \TR\label{item-goal1}, and \item map the transfer of energy through the \TR\ 
with emphasis on \MHD\ waves\label{item-goal2}. \end{inparaenum}
These objectives have significant overlap with the missions of \IRIS~\citep{IRIS14}, the \EIS~\citep{Culhane07}
aboard Hinode, the \EUNIS~\citep{Brosius07,Brosius14}, and a long history of \FUV\ and \EUV\ slit spectrographs.
The \ESIS\ instrument, however, can obtain both spatial and spectral information co-temporally.
This will allow us to resolve complicated morphologies of compact \TR\ reconnection events (as was done with 
\MOSES~\citep{Fox11,Rust17,Courrier18}) and observe signatures of \MHD\ waves over a large portion of the solar disk.
Therefore, in support of goal~\ref{item-goal1}, we will use \ESIS\ to map flows as a function of time and 
space in multiple \TR\ reconnection events.
To achieve goal~\ref{item-goal2}, we will cross-correlate the evolution at multiple temperatures in the \TR\ to map the 
vertical transport of energy over a wide \FOV."""
# In the latest configuration, the \MOSES\ optics are optimized around Ne\,\textsc{vii} (\SI{0.5}{\mega\kelvin}).
# To achieve our goals, \ESIS\ should have a complementary wavelength choice such that we can observe a reasonably
# isolated emission line formed in the lower \TR."""
    )
    result.append(magnetic_reconnection_events.subsection())
    result.append(energy_transfer.subsection())
    result.append(requirements.subsection())
    return result
