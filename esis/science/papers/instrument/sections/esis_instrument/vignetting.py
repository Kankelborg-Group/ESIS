import pylatex
from ... import figures

__all__ = ['subsection']


def subsection() -> pylatex.Subsection:
    result = pylatex.Subsection('Vignetting')
    result.escape = False
    result.append(
        r"""
The original design of \ESIS\ had no vignetting thanks to an stop placed at the primary mirror that was designed to 
perfectly fill the grating with the same amount of light for each point in the \FOV.
This is the \ESIS\ design that was used for the optimization procedure of the grating parameters described in 
Section~\ref{subsec:OptimizationandTolerancing}, for example.
All other results described in the paper use the fully-open system.
Before flight, we decided to remove the primary aperture stop to increase the sensitivity of the instrument at the
expense of introducing vignetting to the \ESIS\ \FOV.
This was acceptable since the vignetting was found to be a simple linear field as shown in Figure~\ref{fig:vignetting},
and could be removed in the post-processing phase."""
    )
    result.append(figures.vignetting.figure())
    return result
