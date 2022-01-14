import pylatex
from ... import tables

__all__ = [
    'subsection',
]


def subsection(doc: pylatex.Document) -> pylatex.Subsection:
    result = pylatex.Subsection('Sensitivity and Cadence')
    result.escape = False
    result.append(
        r"""
Count rates for \ESIS\ are estimated using the expected component throughput from Section~\ref{
subsec:CoatingsandFilters} and the \CCD\ \QE\ listed in Table~\ref{table:prescription}. Line intensities are derived 
from \citet{Vernazza78} (V\&R) \roy{\VR} and the \SOHO/\CDS\ \citep{Harrison95} data, and are given in a variety of 
solar contexts: \QS, \CHs, and \ARs. The \SI{100}{\percent} duty cycle of \ESIS\ (\S\,\ref{subsec:Cameras}) gives us 
the flexibility to use the shortest exposures that are scientifically useful. So long as the shot noise dominates 
over read noise (which is true even for our coronal hole estimates at \SI{10}{\second} exposure length), we can stack 
exposures without a significant \SNR\ penalty. Table~\ref{table:count} shows that \ESIS\ is effectively shot noise 
limited with a \SI{10}{\second} exposure. The signal requirement in Table~\ref{table:scireq} is met by stacking 
exposures. Good quality images ($\sim300$ counts) in active regions can be obtained by stacking \SI{30}{\second} 
worth of exposures. This cadence is sufficient to observe explosive events, but will not resolve torsional Alfv\'en 
waves described in \S\,\ref{sec:ScienceObjectives}. However, by stacking multiple \SI{10}{\second} exposures, 
sufficient \SNR\ \emph{and} temporal resolution of torsional Alfv\'en wave oscillations can be obtained. \roy{Just 
delete these next three sentences?} \jake{Assuming the table and sentences above have been updated to reflect the 
vignetted system, yes}. We also note that the count rates given here are for an unvignetted system which is limited 
by the baffling of this design. While not explored here, there is the possibility of modifying the instrument 
baffling (\S\,\ref{subsec:AperturesandBaffles}) to increase throughput. Thus, a faster exposure cadence may be 
obtained by accepting some vignetting in the system. 
"""
    )
    # result.append(tables.counts.table_old)
    result.append(tables.counts.table(doc))
    return result
