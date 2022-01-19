import pylatex
from ... import figures

__all__ = [
    'subsection'
]


def subsection() -> pylatex.Subsection:
    result = pylatex.Subsection(pylatex.NoEscape(r'Limitations of the \MOSES\ Design'))
    result.escape = False
    result.append(r"""
The \MOSES\ design features a single concave diffraction grating forming images on three \CCD\ 
detectors~\citep{Fox10} (Figure~\ref{fig:mosesSchematic}). 
The optical path is folded in half by a single flat secondary mirror (omitted in Figure~\ref{fig:mosesSchematic}).
Provided that the three cameras are positioned correctly, this arrangement allows the entire telescope to be brought 
into focus using only the central (undispersed) order and a visible light source.
Unfortunately this design uses volume inefficiently for two reasons.
First, the lack of magnification by the secondary mirror limits the folded length of the entire telescope to be no less 
than half of the \SI{5}{\meter} focal length of the grating~\citep{Fox10,Fox11}.
Second, the dispersion of the instrument is controlled by the placement of the cameras.
To achieve the maximum dispersion of \SI{29}{\kilo\meter\per\second}~\citep{Fox10}, the outboard orders are imaged as 
far apart as possible in the $\sim\text{\skinDiameter}$ diameter envelope of the rocket payload.
The resulting planar dispersion poorly fills the cylindrical volume of the payload, leaving much unused space along the 
orthogonal planes."""
    )

    result.append(figures.schematic_moses.figure())

    result.append(r"""
Furthermore, the monolithic secondary, though it confers the focus advantage noted above, does not 
allow efficient placement of the $m=\pm1$ \CCDs.  
For all practical purposes, the diameter of the payload (\skinDiameter) can only accommodate three diffraction 
orders ($m=-1, 0, +1$).
Therefore, \textit{\MOSES\ can only collect, at most, three pieces of information at each point in the field of view.}
From this, it is not reasonable to expect the reconstruction of more than three degrees of freedom for each spectral line, 
except in the case very compact, isolated features such as those described by \citet{Fox10} and \citet{Rust17}.
Consequently, it is a reasonable approximation to say that \MOSES\ is sensitive primarily to spectral line intensities, 
shifts, and widths \citep{KankThom01}.
With any tomographic apparatus, the degree of detail that can be resolved in the object depends critically on the 
number of viewing angles~\citep{Kak88,Descour97,Hagen08}.
So it is with the spectrum we observe with \MOSES: more dispersed images are required to confer sensitivity to finer 
spectral details such as higher moments of the spectral line shape.

A related issue stems from the use of a single grating, with a single dispersion plane.
Since the solar corona and transition region are structured by magnetic fields, the scene tends to be dominated by 
\sout{field aligned} \roy{field-aligned} structures such as loops~\citep{Rosner78,Bonnet80}.
When the \MOSES\ dispersion direction happens to be aligned nearly perpendicular to the magnetic field, filamentary 
structures on the transition region serve almost as spectrograph slits unto themselves.
The estimation of Doppler shifts then becomes a simple act of triangulation, and broadenings are also readily 
diagnosed~\citep{Fox10,Courrier18}.
A double-peaked profile can also be observed with sufficiently isolated features~\citep{Rust17}.
Unfortunately, solar magnetic fields in the transition region are quite complex and do not have a global preferred 
direction.
In cases where the field is nearly parallel to the instrument dispersion, spectral shifts and broadenings are not 
readily apparent.

The single diffraction grating also leads to a compromise in the optical performance of the instrument.
Since the \MOSES\ grating forms images in three orders simultaneously, \sout{aberration cannot be simultaneously optimized for} 
\roy{there aren't enough degrees of freedom in the optical system to achieve sub-pixel aberrations in} all three of 
those spectral orders. 
A result of this design is that the orientations (\ie\,the major axis) of the \PSF\ varies 
order to order~\citep{Rust17}. 
During the first mission, \MOSES\ was flown with a small amount of defocus~\citep{Rust17}, which exacerbated the 
inter-order \PSF\ variation and caused the individual \PSFs\ to span several 
pixels~\citep{Rust17,Atwood18}. 
The combination of these two effects results in spurious spectral features that 
require additional consideration~\citep{Atwood18} and further increase the complexity of the inversion 
process~\citep{Rust17,Courrier18}. 

\jake{Another complication with \MOSES\ is that the image in each spectral order contains a different combination of 
spatial and spectral information.  
This stems from the fact that \MOSES\ lacks a field stop to define a wavelength independent \FOV\ and that it uses an 
undispersed channel. In this configuration, intensity from wavelengths off the primary observing wavelength, 
but within the \MOSES\ passband, will be imaged in the zero order, but may be dispersed off the detector in the outboard orders. 
In the opposite sense, features outside of the \FOV\ of the zeroth order may be dispersed onto either of the outboard 
order detectors. \citet{Parker2022} compared synthetic \MOSES\ images to the real data and found that approximately 
ten percent of the intensity in the zeroth order image originated from more than ten dims lines in the \MOSES\ passband, 
most of which are too dim to be visible in the dispersed images.  
This study revealed that undispered channels, although attractive due to their lack of spatial-spectral ambiguity, 
can provide misleading intensity information which limits their utility in inversion without careful forward modeling.  
Also, that the \FOV\ should be clearly defined, and the same, for each wavelength so that the spectral contribution to a 
given pixel is clearly defined.} 

Finally, the exposure cadence of \MOSES\ is hindered by an $\sim$\SI{6}{\second} readout time for the \CCDs~\citep{
Fox11}. 
The observing interval for a solar sounding rocket flight is very short, typically about five minutes. 
Consequently, every second of observing time is precious, both to achieve adequate exposure time and to catch the 
full development of dynamical phenomena. 
The \MOSES\ observing duty cycle is $\sim$\SI{50}{\percent} since it is limited by the readout time of its \CCDs. 
Thus, valuable observing time is lost. The readout data gap compelled us to develop a \MOSES\ exposure sequence with 
exposures ranging from $0.25$-\SI{24}{\second}, a careful trade-off between deep and fast exposures. 

In summary, our experience leads us to conclude that the \MOSES\ design has the following primary limitations:
\begin{enumerate}
    \item inefficient use of volume \label{item-length} %(x and y direction)
    \item dispersion constrained by payload dimensions \label{item-disp_con}
    \item too few dispersed images (orders) \label{item-orders}
    \item single dispersion plane \label{item-dispersion}
    \item insufficient degrees of freedom to control aberrations \label{item-PSF}
    \item poorly-defined and wavelength dependent \FOV\ \label{item-FOV}
    \item misleading intensity in $m=0$ order image
    \item low duty cycle \label{item-CAD}
\end{enumerate}
In designing \ESIS, we have sought to improve upon each of these points."""
    )

    return result
