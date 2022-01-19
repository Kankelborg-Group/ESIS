import pylatex
from ... import figures

__all__ = [
    'subsection'
]


def subsection() -> pylatex.Subsection:
    result = pylatex.Subsection('Apertures and Baffles')
    result.escape = False
    result.append(figures.baffles.figure())
    result.append(
        r"""
\jake{Needs a big rework}               
Each channel of \ESIS\ has two apertures: one at the surface of the grating and another in front of the primary mirror.
The purpose of the aperture at the grating is to mask the out-of-figure margins at the edges of these optics.
This provides a well defined edge to the clear aperture of each grating while also keeping unwanted rays from being 
reflected from the grating margins and back onto the \CCDs.
The dimensions of the grating aperture match those of the grating clear aperture shown in Figure~\ref{fig:schematic}c. 

The aperture placed at the primary mirror is the stop for each individual channel.
The area of the stop has been maximized under the constraint that no rays be vignetted anywhere else in the system.
The gratings and their clear apertures were the most significant areas of concern for potential vignetting.
Thus, the shape of stop at the primary is largely influenced by the shape of the grating clear aperture.
The inner extent of the primary stop (the ``tip'' of the triangle in Figure~\ref{fig:schematic}b) is defined by the 
occultation of the primary by the shadow cast from the gratings and their mounts.
This presented an intricate geometry problem, as the occultation is a function of the incoming field angle, 
the radial extent of the grating mount, and the distance of the mount to the primary mirror along the optical axis.
Hence, the inner extent of the primary stop was solved for iteratively with the optimization described in 
Section~\ref{subsec:OptimizationandTolerancing}, which affected the placement of the gratings relative to the primary mirror.
The resulting optimized and non-vignetting stop geometry is shown in Figure~\ref{fig:schematic}b.

After final optimization, the stop geometry was analyzed to check for vignetting at the grating with the optical model.
A footprint diagram was generated at the grating from of multiple grids of rays.
The incidence angle of each grid of rays corresponded to the extremes of \FOV\ defined by the positions of the eight 
points of the octagonal field stop.
The footprint diagram showed that, with the stop completely filled, no ray landed outside of the grating clear aperture
in Figure~\ref{fig:schematic}c, and no ray was intercepted by the central obscuration.

From Figure~\ref{fig:schematic}c it is apparent that considerable surface area of the primary mirror is unused by the 
non-vignetting stop design.
The primary apertures could be enlarged considerably if the vignetting constraint were to be relaxed.

The \ESIS\ baffles are designed to block direct light paths between the front aperture plate and the \CCDs\ for any ray 
$<$\SI{1.4}{\degree} from the optical axis.
This angle is purposefully larger than the angular diameter of the sun ($\sim$\SI{0.5}{\degree}) so that any direct 
paths are excluded from bright sources in the solar corona.
All baffles are bead-blasted, anodized \Al\ sheet metal oriented perpendicular to the optical axis.
The size and shape of the cutouts were determined using a combination of the ray trace from 
Section~\ref{subsec:OptimizationandTolerancing} and 3D modeling.
The light path from the primary mirror to the field stop is defined as the volume that connects each vertex of the 
primary mirror aperture mask (e.g., Fig.~\ref{fig:schematic}) to every vertex of the octagonal field stop.
This is a conservative definition that ensures no rays within the \FOV\ are excluded, and therefore unintentionally 
vignetted by the baffles.  Light paths from the field stop to the grating, and from the grating to the image formed on 
the \CCD, are defined in a similar manner.
The cutouts in the baffles are sized using the projection of these light paths onto the baffle surface.
A conservative \SI{1}{\milli\meter} margin is added to each cutout to prevent unintentional vignetting.
A model of the six baffles, showing cutouts and position on the optical bench, is displayed in Fig.~\ref{F-Baff1}."""
    )
    return result
