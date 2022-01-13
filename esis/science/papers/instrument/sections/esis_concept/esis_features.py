import pylatex
from ... import figures

__all__ = [
    'subsection'
]


def subsection() -> pylatex.Subsection:
    result = pylatex.Subsection(pylatex.NoEscape(r'\ESIS\ Features'))
    result.escape = False
    result.append(figures.layout.figure())
    result.append(
        r"""
The layout of \ESIS\ (Figure~\ref{fig:layout}) is a modified form of Gregorian telescope.
Incoming light is brought to focus at an octagonal field stop by a parabolic primary mirror.
In the \ESIS\ layout, the secondary mirror of a typical Gregorian telescope is replaced by a segmented, octagonal array 
of diffraction gratings.
From the field stop, the gratings re-image to \CCD\ detectors arranged radially around the primary mirror.
The gratings are blazed for first order, so that each \CCD\ is fed by a single corresponding grating, and all the 
gratings are identical in design.
The features of this new layout address all of the limitations described in 
Section~\ref{subsec:LimitationsoftheMOSESDesign}, and are summarized here.

Replacing the secondary mirror with an array of concave diffraction gratings confers several advantages to \ESIS\ 
over \MOSES. 
First, the concavity of the gratings creates magnification in the \ESIS\ optical system, which results in a shorter axial 
length than \MOSES, without sacrificing spatial or spectral resolution. 
Second, the magnification and tilt of an individual grating controls the position of the dispersed image with respect to 
the optical axis, so that the spectral resolution is not as constrained by the payload dimensions. 
Third, the radial symmetry of the design places the cameras closer together, resulting in a more compact instrument. 
Furthermore, by arranging the detectors around the optical axis, more dispersed grating orders can be populated; up to 
eight gratings can be arrayed around the \ESIS\ primary mirror (up to six with the current optical table). 
This contrasts the three image orders available in the planar symmetry of \MOSES. 
Taken together, these three design features make \ESIS\ more compact than \MOSES\ \sout{(\S\,\ref{subsec:LimitationsoftheMOSESDesign} 
item~\ref{item-length})} \roy{(Limitation~\ref{item-length})}, improve spectral resolution \sout{(\S\,
\ref{subsec:LimitationsoftheMOSESDesign} item~\ref{item-disp_con})} \roy{(Limitation~\ref{item-disp_con})} and allow 
the collection of more projections to better constrain the interpretation of the data \sout{(\S\,
\ref{subsec:LimitationsoftheMOSESDesign} item~\ref{item-orders})} \roy{(Limitation~\ref{item-orders})}. 
 
The \ESIS\ gratings are arranged in a segmented array, clocked in \SI{45}{\degree} increments, so that there are 
\numChannelsWords\ distinct dispersion planes. 
This will greatly aid in reconstructing spectral line profiles since the dispersion space of \ESIS\ occupies a 
3D volume rather than a 2D plane as with \MOSES. For \ESIS, there will always be a dispersion plane within 
\SI{22.5}{\degree} of the normal to any loop-like feature in the solar atmosphere. 
As discussed in Section~\ref{subsec:LimitationsoftheMOSESDesign}, a nearly perpendicular dispersion plane 
allows a filamentary structure to serve like a spectrographic slit, resulting in a clear presentation of the 
spectrum. 
This feature addresses \sout{\S\,\ref{subsec:LimitationsoftheMOSESDesign} item~\ref{item-dispersion}} \roy{
Limitation~\ref{item-dispersion}}. 

Rather than forming images at three spectral orders from a single grating, each \ESIS\ imaging channel has a 
dedicated grating. 
Aberrations are controlled by optimizing the grating design to form images in first order, 
over a narrow range of ray deviation angles. 
This design controls aberration well enough to allow pixel-limited imaging, avoiding the \PSF\ mismatch problems 
inherent to the \MOSES\ design (\S\,\ref{subsec:LimitationsoftheMOSESDesign} item \ref{item-PSF}). 
In its flight configuration with gratings optimized around a \OVwavelength\ wavelength, the instrument cannot be aligned and 
focused in visible light like \MOSES. 
Visible gratings and an special alignment transfer procedure(\S\,\ref{subsec:AlignmentandFocus}) must be used for the 
alignment and focus of \ESIS. 

The \ESIS\ design also includes an octagonal field stop placed at prime focus.
This confers two advantages.
First, the field stop fully defines the instrument \FOV, so that \ESIS\ is not susceptible to the spectral confusion 
observed in \MOSES\ data (\S\,\ref{subsec:LimitationsoftheMOSESDesign} limitation~\ref{item-FOV}).
Second, each spectral image observed by \ESIS\ will be bordered by the outline of the field stop 
(\eg\,\S\,\ref{subsec:Optics}).
This aids the inversion process since outside of this sharp edge the intensity is zero for any look angle through an 
\ESIS\ data cube.
Additionally, the symmetry of the field stop gives multiple checkpoints where the edge inversion is duplicated in the 
dispersed images produced by adjacent orders.
The size and octagonal shape of the field stop are defined by the requirement that all \CCDs\ must see the entire \FOV\ 
from edge to edge, while leaving a small margin for alignment. 

Lastly, in contrast to \MOSES, \ESIS\ employs frame transfer \CCDs\ to make optimum use of our five minutes of observing 
time.
The \ESIS\ design is shutterless, so that each detector is always integrating.
The result is a \SI{100}{\percent} duty cycle.
The lack of downtime for readout also allows \ESIS\ to operate at a fixed, rapid cadence of $\sim$\SI{3}{\second}.
Longer integration times can be achieved for faint features by exposure stacking 
(\S\,\ref{subsec:LimitationsoftheMOSESDesign} item~\ref{item-CAD}).

In summary, the \ESIS\ concept addresses all the limitations of the \MOSES\ design enumerated in 
\S\,\ref{subsec:LimitationsoftheMOSESDesign}.
The volume of the \ESIS\ optical layout is smaller than \MOSES\ by almost a factor of two, yet with a smaller \PSF, 
improved spectral resolution, and faster exposure cadence.
\ESIS\ offers several features to improve the recovery of spectral information, including more channels, crossed 
dispersion planes, and a field stop."""
    )
    return result
