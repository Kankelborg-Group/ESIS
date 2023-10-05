import pylatex

__all__ = [
    'section',
]


def section() -> pylatex.Section:
    result = pylatex.Section('Introduction')
    result.escape = False
    result.append(
        r"""
The solar \TR\ and corona, as viewed from space in its characteristic short wavelengths (\FUV, \EUV, and soft X-ray), 
is a three-dimensional scene evolving in time:  $I(x, y, \lambda, t)$.
Here, the helioprojective cartesian coordinates, $x$ and $y$ \citep{Thompson2006}, and the wavelength axis, $\lambda$, 
comprise the three dimensions of the scene, while $t$ represents the temporal axis.
An ideal instrument would capture a spatial/spectral data cube, at a rapid temporal cadence, however, practical 
limitations lead us to accept various compromises of the sampling rate along each of these four dimensions.
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
The proposed \MUSE\ \citep{DePontieu2020} mission addresses this problem in yet another way by using 
37 slits to obtain simultaneous spectra over a 2D \FOV.
With the multiple slits, it requires only an 11 step raster to acquire a dense, $170'' \times 170''$ spatial/spectral 
cube at $0.8'' \times 0.4''$ resolution.
\MUSE can acquire such a raster in $\sim\qtyrange{2}{4}{\minute}$.

A different and radical approach is to use a \textit{slitless spectrograph}.
Traditionally, imaging spectrographs use a slit to provide an unambiguous spectrum as a function of a single spatial 
dimension $y$, along the slit. 
Without a slit, spatial variations are obtained in the orthogonal spatial dimension, $x$, at the cost of degeneracy 
between $x$ and $\lambda$.
In \EUV, where we find negligible continuum, the grating forms a series of overlapping images in the strong emission lines. 
Such an image may be described as an \textit{overlappogram}.
The \Acposs{NRL} S082A spectroheliograph~\citep{Tousey73,Tousey77}  was one of the first instruments to pioneer this method.
The overlappograms obtained by S082A identified hundreds of spectral line transitions~\citep{Feldman85}, and have more 
recently been used to determine line ratios in solar flares~\citep{Keenan06}.
Unfortunately, for closely-spaced \EUV\ lines, the dispersed images from the single diffraction order suffer from 
considerable ambiguity from overlapping images.
Image overlap is all but unavoidable with this configuration, however, overlappograms can be disentangled, or inverted, 
to recover the spatial/spectral cube under the right circumstances.  

In analogy to a tomographic imaging problem~\citep{Kak88}, inversion of an overlapping spatial/spectral scene can be 
facilitated by increasing the number of independent spectral projections, or viewing angles, through the 3D 
$(x,y,\lambda)$ scene~\citep{Descour97}.
The quality of the inversion (\eg\ recovery of higher order spectral line moments) can be improved by additional 
projections~\citep{Kak88,Descour97}, generally at the cost of computational complexity~\citep{Hagen08}.
\Acp{CTIS}~\citep{okamoto1991,Bulygin91,Descour95} leverage this concept by obtaining multiple, simultaneous dispersed 
images of an object or scene; 
upwards of 25 grating diffraction orders may be projected onto a single detector plane~\citep{Descour97}.
Through post-processing of these images, \CTISs\ can recover a 3D data cube from a (spectrally) smooth and continuous 
scene over a large bandpass (\eg\ \citet{Hagen08}).
A very similar technique, operating at much higher spectral resolution, was independently pioneered for solar physics by 
\citet{DeForest04}, who demonstrated synthesis of a magnetogram from a single exposure using two dispersed spectral 
orders of a grating at the Dunn Solar Telescope.

The \MOSES~\citep{Fox10,Fox11} demonstrated the unique capability of simultaneous imaging and spectroscopy for 
solar \EUV\ observations.
\MOSES\ is a three-order slitless spectrograph that seeks to combine the simplicity of the S082A concept with the 
advantages of a \CTIS\ instrument.
A single diffraction grating (in conjunction with a fold mirror) projects the $m=\pm1$ and the undispersed  $m=0$ order 
onto three different detectors.
Through a combination of dispersion and multi-layer coatings, the passband of the $m=\pm1$ orders encompasses only a few
solar \EUV\ emission lines.
The resulting spareseness of the 3D data cube helps make inversion of \MOSES\ data better-posed.
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
