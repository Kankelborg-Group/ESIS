import pylatex

__all__ = [
    'section',
]


def section() -> pylatex.Section:
    result = pylatex.Section('Introduction')
    result.escape = False
    result.append(
        r"""
The solar atmosphere, as viewed from space in its characteristic short wavelengths (\FUV, \EUV, and soft X-ray), is a 
three-dimensional scene evolving in time:  $I(x, y, \lambda, t)$.
Here, the helioprojective cartesian coordinates, $x$ and $y$ \citep{Thompson2006}, and the wavelength axis, $\lambda$, 
comprise the three dimensions of the scene, while $t$ represents the temporal axis.
An ideal instrument would capture a spatial/spectral data cube, $I(x, y, \lambda)$, at a rapid temporal cadence, 
however, practical limitations lead us to accept various compromises of the sampling rate along each of these four dimensions.
Approaching this ideal is the fast tunable filtergraph (\ie\ fast tunable Fabry--P\'erot etalons, \eg\ the GREGOR 
Fabry--P{\'e}rot Interferometer, \citep{Puschmann12}), but the materials do not exist to extend this technology to 
\EUV\ wavelengths shortward of $\sim$\SI{150}{\nano\meter}~\citep{2000WuelserFP}.
Imagers like the \TRACE~\citep{Handy99} and the \AIA~\citep{Lemen12} collect 
narrowband \EUV\ images of the solar atmosphere at high cadence, but their passbands are too wide to isolate a single emission line.
In principle, filter ratios that make use of spectrally-adjacent multilayer \EUV\ passbands could detect Doppler 
shifts~\citep{Sakao99}.
However, the passbands of the multilayer coatings are still wide enough that the presence of weaker contaminant lines 
limits resolution of Doppler shifts to $\sim$\SI{1000}{\kilo\meter\per\second}~\citep{Kobayashi00}.
Slit spectrographs (\eg\ the \IRIS~\citep{IRIS14}) obtain fast, high-resolution spatial and spectral observations, but are 
limited by the narrow \FOV\ of the spectrograph slit.
The $I(x, y, \lambda)$ data cube can be built up by rastering the slit pointing, but it cannot be co-temporal along the 
raster axis.
Moreover, extended and dynamic scenes can change significantly in the time required to raster over their extent.  

\roy{
A different approach is to use a \textit{slitless spectrograph}, a spectrograph built without the slit employed by 
traditional spectrographs.
This is a radical approach because the slit was there so the spectrum could be interpreted unambiguously.
Without the slit, the spectral direction and the spatial direction that would have been perpendicular to the slit are 
degenerate.
However, the solar atmosphere viewed in \EUV\ is a perfect candidate for observation using a slitless spectrograph since 
the spectrum is dominated by emission lines, and has low continuum.
So instead of a smear, a slitless spectrograph observing the Sun in \EUV\ would capture an image of many overlapping and shifted
copies of the Sun, one for each spectral line in the passband, this type of image is known as an \textit{overlappogram}.}
The \Acposs{NRL} SO82A spectroheliograph~\citep{Tousey73,Tousey77}  was one of the first instruments to pioneer this method.
The overlappograms obtained by SO82A identified several spectral line transitions~\citep{Feldman85}, and have more 
recently been used to determine line ratios in solar flares~\citep{Keenan06}.
Unfortunately, for closely-spaced \EUV\ lines, the dispersed images from the single diffraction order suffer from 
considerable ambiguity from overlapping images.
Image overlap is all but unavoidable with this configuration, however, overlappograms can be disentangled, or inverted, 
to recover the spatial/spectral cube under the right circumstances.  

In analogy to a tomographic imaging problem~\citep{Kak88}, inversion of an overlapping spatial/spectral scene can be 
facilitated by increasing the number of independent spectral projections, or viewing angles, through the 3D 
$(x,y,\lambda)$ scene~\citep{DeForest04}.
For example, \citet{DeForest04} demonstrated recovery of Doppler shifts in magnetograms from two dispersed orders of a 
grating at the output of the \MDI~\citep{Scherrer95}.
The quality of the inversion (\eg\ recovery of higher order spectral line moments) can also be improved by additional 
projections~\citep{Kak88,Descour97}, generally at the cost of computational complexity~\citep{Hagen08}.
\Acp{CTIS}~\citep{okamoto1991,Bulygin91,Descour95} leverage this concept by obtaining multiple, simultaneous dispersed 
images of an object or scene; 
upwards of 25 grating diffraction orders may be projected onto a single detector plane~\citep{Descour97}.
Through post-processing of these images, \CTISs\ can recover a 3D data cube from a (spectrally) smooth and continuous 
scene over a large bandpass (\eg\ \citet{Hagen08}).

The \jake{capitalize "O"?}\MOSES~\citep{Fox10,Fox11} is our first effort aimed at developing the unique capability of simultaneous 
imaging and spectroscopy for solar \EUV\ scenes.
\MOSES\ is a three-order slitless spectrograph that seeks to combine the simplicity of the SO82A concept with the 
advantages of a \CTIS\ instrument.
A single diffraction grating (in conjunction with a fold mirror) projects the $m=\pm1$ and the undispersed  $m=0$ order 
onto three different detectors.
Through a combination of dispersion and multi-layer coatings, the passband of the $m=\pm1$ orders encompasses only a few
solar \EUV\ emission lines.
Thus, \MOSES\ overlappograms consist of only a handful of spectral images.
This constraint on the volume of the 3D data cube helps make inversion of \MOSES\ data better-posed despite the 
discontinuous nature of the solar \EUV\ spectrum.
This working concept enabled by \MOSES\ has been proven over the course of two previous rocket flights.
Through inversion of \MOSES\ overlappograms, \citet{Fox10} obtained unprecedented measurements of Doppler shifts 
(\ie\ line widths) of \TR\ explosive events as a function of time and space while \citet{Rust17} recovered splitting and 
distinct moments of compact \TR\ bright point line profiles.

\jake{might just need to go through the whole paper then write this paragraph}
Building on the working concept demonstrated by \MOSES, here we describe a new instrument, the \ESIS, that improves
on past efforts to perform simultaneous imaging and spectroscopy in \EUV.
In Section~\ref{sec:TheESISConcept} we detail how our experience with the \MOSES\ instrument has shaped the design of 
\ESIS.
Section~\ref{sec:ScienceObjectives} describes the narrow scientific objectives and the requirements placed on the new 
instrument.
Section~\ref{sec:TheESISInstrument} describes the technical approach to meet our scientific objectives, followed by a 
brief description of the expected mission profile in Section~\ref{sec:MissionProfile}."""
    )
    return result
