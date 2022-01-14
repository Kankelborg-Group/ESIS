import pylatex
from ... import figures

__all__ = [
    'subsection'
]


def subsection() -> pylatex.Subsection:
    result = pylatex.Subsection('Optics')
    result.escape = False
    result.append(
        r"""
Figure~\ref{fig:schematic}a shows the relative layout of the optics and detectors for a single 
\ESIS\ channel.
Here we give specific details of the primary mirror and gratings (Fig.~\ref{fig:schematic}b and \ref{fig:schematic}c, respectively).
The features of the field stop have been described previously in Sec.~\ref{subsec:ESISFeatures}, while the \CCD\ and 
cameras are covered in Sec.~\ref{subsec:Cameras}. """
    )
    result.append(figures.schematic.figure())
    result.append(
        r"""
The primary mirror is octagonal in shape.
The octagonal shape of the primary allows dynamic clearance for filter tubes that are arranged radially around the 
mirror (\S\,\ref{subsec:CoatingsandFilters}).
The mirror is attached to a backing plate by three \textit{bipods}: thin titanium structures that are flexible in the radial 
dimension, perpendicular to the mirror edge, but rigid in the other two dimensions.
The bipods form a kinematic mount, isolating the primary mirror figure from mounting stress. 

The mirror will have to maintain its figure under direct solar illumination, so a Corning \ULE\ substrate was used.
The transparency of \ULE, in conjunction with the transparency of the mirror coating in visible and near-IR  wavelengths 
(\eg, Table~\ref{table:prescription} and \S\,\ref{subsec:CoatingsandFilters}), helps minimize 
the heating of the mirror.
Surface figure specifications for the \ESIS\ optics are described in Sec.~\ref{subsec:OptimizationandTolerancing}.

The spherical gratings (Fig.~\ref{fig:schematic}c) re-image light from the field stop to form dispersed images at the 
\CCDs.
Each grating is individually mounted to a backing plate in a similar fashion as the primary mirror.
For these much smaller optics, lightweight bipods were photo-chemically etched from thin titanium sheet.
The bipods are bonded to both the grating and backing plate along the three long edges of each grating.
The individual mounts allow each grating to be adjusted in tip and tilt to center the image on the \CCD. """
    )
    result.append(figures.field_stop_projections.figure())
    result.append(
        r"""
The gratings have a varied line space ruling pattern optimized to provide, in principle, 
pixel-limited imaging from the field stop to the \CCDs.
The pitch at the center of the grating is $d_0=\text{\gratingRulingSpacing}$ resulting in a dispersion of 
\dispersionDoppler\ at the center of the \OV\ \FOV.
The groove profile is optimized for the $m=1$ order, so that each grating serves only a single \CCD.
The modeled grating groove efficiency in this order is \SI{36}{\percent} \roy{We said \SI{39}{\percent} above, need to 
find out which it is, I get \gratingGrooveEfficiency} at \OV. 

Figure specification and groove profile are not well controlled near the edges of the gratings. Therefore, 
\jake{an uncoated section of mirror was left around the edge of the grating when applying the multilayer coating, 
minimizing reflection in EUV.} Fig.~\ref{fig:schematic}c 

The \ESIS\ passband is defined through a combination of the field stop, the grating dispersion, and the \CCD\ size.
The passband includes the \HeI\ spectral line through \MgXion\ (\MgXwavelength\ and \MgXdimWavelength) to \OV.
Figure~\ref{fig:projections} shows where images of each of the strong spectral lines will fall on the \CCD.
The instrument dispersion satisfies the spectral resolution requirement in Table~\ref{table:scireq} and ensures that the 
spectral images are well-separated; Figure~\ref{fig:projections} shows that \HeI\ will be completely 
separated from the target \OV\ line."""
    )
    return result
