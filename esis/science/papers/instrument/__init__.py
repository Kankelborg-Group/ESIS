import pathlib
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
import pylatex
import kgpy.latex
import kgpy.units
import kgpy.chianti
import esis.optics
import esis.science.papers.instrument.figures as figures

path_base = pathlib.Path(__file__).parent
path_pdf = path_base / 'instrument'
path_figures = path_base / 'figures'


def document() -> kgpy.latex.Document:
    doc = kgpy.latex.Document(
        default_filepath=str(path_pdf),
        documentclass='aastex631',
        document_options=[
            'twocolumn',
            # 'linenumbers',
        ]
    )

    doc.packages.append(pylatex.Package('paralist'))
    doc.packages.append(pylatex.Package('acronym'))
    doc.packages.append(pylatex.Package('savesym'))
    doc.preamble.append(pylatex.NoEscape(
        '\\savesymbol{tablenum}\n'
        '\\usepackage{siunitx}\n'
        '\\restoresymbol{SIX}{tablenum}\n'
    ))

    doc.preamble.append(pylatex.NoEscape(
        r"""
\makeatletter
\newcommand{\acposs}[1]{%
 \expandafter\ifx\csname AC@#1\endcsname\AC@used
   \acs{#1}'s%
 \else
   \aclu{#1}'s (\acs{#1}'s)%
 \fi
}
\newcommand{\Acposs}[1]{%
 \expandafter\ifx\csname AC@#1\endcsname\AC@used
   \acs{#1}'s%
 \else
   \Aclu{#1}'s (\acs{#1}'s)%
 \fi
}
\makeatother"""
    ))

    doc.set_variable(
        name='ie',
        value=pylatex.NoEscape(r'\textit{i.e.}')
    )

    doc.set_variable(
        name='eg',
        value=pylatex.NoEscape(r'\textit{e.g.}')
    )

    doc.preamble.append(pylatex.NoEscape(r'\newcommand{\amy}[1]{{{\color{red} #1}}}'))
    doc.preamble.append(pylatex.NoEscape(r'\newcommand{\jake}[1]{{{\color{purple} #1}}}'))
    doc.preamble.append(pylatex.NoEscape(r'\newcommand{\roy}[1]{{{\color{blue} #1}}}'))

    doc.preamble.append(pylatex.Command('bibliographystyle', 'aasjournal'))
    doc.set_variable('spiejatis', pylatex.NoEscape(r'J~.Atmos.~Tel. \& Img.~Sys.'))

    doc.append(kgpy.latex.Title('The EUV Snapshot Imaging Spectrograph'))

    affil_msu = kgpy.latex.aas.Affiliation(
        'Montana State University, Department of Physics, '
        'P.O. Box 173840, Bozeman, MT 59717, USA'
    )

    affil_msfc = kgpy.latex.aas.Affiliation(
        'NASA Marshall Space Flight Center, '
        'Huntsville, AL 35812, USA'
    )

    affil_lbnl = kgpy.latex.aas.Affiliation(
        'Lawrence Berkeley National Laboratory, '
        '1 Cyclotron Road, Berkeley, CA 94720, USA'
    )

    affil_rxo = kgpy.latex.aas.Affiliation(
        'Reflective X-ray Optics LLC, '
        '425 Riverside Dr., #16G, New York, NY 10025, USA'
    )

    doc.append(kgpy.latex.aas.Author('Hans T. Courrier', affil_msu))
    doc.append(kgpy.latex.aas.Author('Roy T. Smart', affil_msu))
    doc.append(kgpy.latex.aas.Author('Charles C. Kankelborg', affil_msu))
    doc.append(kgpy.latex.aas.Author('Amy R. Winebarger', affil_msfc))
    doc.append(kgpy.latex.aas.Author('Ken Kobayashi', affil_msfc))
    doc.append(kgpy.latex.aas.Author('Brent Beabout', affil_msfc))
    doc.append(kgpy.latex.aas.Author('Dyana Beabout', affil_msfc))
    doc.append(kgpy.latex.aas.Author('Ben Carrol', affil_msu))
    doc.append(kgpy.latex.aas.Author('Jonathan Cirtain', affil_msfc))
    doc.append(kgpy.latex.aas.Author('James A. Duffy', affil_msfc))
    doc.append(kgpy.latex.aas.Author('Eric Gullikson', affil_lbnl))
    doc.append(kgpy.latex.aas.Author('Micah Johnson', affil_msu))
    doc.append(kgpy.latex.aas.Author('Jacob D. Parker', affil_msu))
    doc.append(kgpy.latex.aas.Author('Laurel Rachmeler', affil_msfc))
    doc.append(kgpy.latex.aas.Author('Larry Springer', affil_msu))
    doc.append(kgpy.latex.aas.Author('David L. Windt', affil_rxo))

    optics_single = esis.optics.design.final(
        pupil_samples=11,
        pupil_is_stratified_random=True,
        field_samples=11,
        field_is_stratified_random=True,
        all_channels=False,
    )

    optics = esis.optics.design.final(
        pupil_samples=11,
        pupil_is_stratified_random=True,
        field_samples=11,
        field_is_stratified_random=True,
    )

    doc.set_variable_quantity(
        name='fov',
        value=optics_single.field_of_view.quantity.mean().to(u.arcmin),
        digits_after_decimal=1,
    )

    doc.set_variable_quantity(
        name='avgPlateScale',
        value=optics_single.plate_scale.quantity.mean(),
        digits_after_decimal=2,
    )

    doc.set_variable_quantity(
        name='minSpatialResolution',
        value=optics_single.resolution_spatial.quantity.min(),
        digits_after_decimal=2,
    )

    doc.set_variable_quantity(
        name='dispersion',
        value=optics_single.dispersion.to(kgpy.units.mAA / u.pix),
        digits_after_decimal=0,
    )

    doc.set_variable_quantity(
        name='dispersionDoppler',
        value=optics_single.dispersion_doppler,
        digits_after_decimal=0,
    )

    doc.set_variable_quantity(
        name='minCadence',
        value=optics_single.detector.exposure_length_min,
        digits_after_decimal=1,
    )

    doc.set_variable_quantity(
        name='primaryDiameter',
        value=2 * optics_single.primary.clear_half_width,
        digits_after_decimal=0,
    )

    wavelength = optics.bunch.wavelength
    ion = kgpy.chianti.to_spectroscopic(optics.bunch.ion)

    index_o5 = np.nonzero(optics.bunch.ion == 'o_5')[0][0]
    doc.set_variable_quantity(
        name='OVwavelength',
        value=wavelength[index_o5],
        digits_after_decimal=3,
    )
    doc.set_variable(
        name='OVion',
        value=pylatex.NoEscape(ion[index_o5])
    )
    doc.set_variable(
        name='OV',
        value=pylatex.NoEscape(r'\OVion\ \OVwavelength')
    )

    index_he1 = np.nonzero(optics.bunch.ion == 'he_1')[0][0]
    doc.set_variable_quantity(
        name='HeIwavelength',
        value=wavelength[index_he1],
        digits_after_decimal=3,
    )
    doc.set_variable(
        name='HeIion',
        value=pylatex.NoEscape(ion[index_he1])
    )
    doc.set_variable(
        name='HeI',
        value=pylatex.NoEscape(r'\HeIion\ \HeIwavelength')
    )

    index_mg10 = np.nonzero(optics.bunch.ion == 'mg_10')[0][0]
    doc.set_variable_quantity(
        name='MgXwavelength',
        value=wavelength[index_mg10],
        digits_after_decimal=3,
    )
    doc.set_variable(
        name='MgXion',
        value=pylatex.NoEscape(ion[index_mg10])
    )
    doc.set_variable(
        name='MgX',
        value=pylatex.NoEscape(r'\MgXion\ \MgXwavelength')
    )

    index_mg10_2 = np.nonzero(optics.bunch.ion == 'mg_10')[0][1]
    doc.set_variable_quantity(
        name='MgXdimWavelength',
        value=wavelength[index_mg10_2],
        digits_after_decimal=3,
    )
    doc.set_variable(
        name='MgXdimIon',
        value=pylatex.NoEscape(ion[index_mg10_2])
    )
    doc.set_variable(
        name='MgXdim',
        value=pylatex.NoEscape(r'\MgXdimIon\ \MgXdimWavelength')
    )

    doc.preamble.append(kgpy.latex.Acronym('ESIS', r'EUV Snapshot Imaging Spectrograph'))
    doc.preamble.append(kgpy.latex.Acronym('MOSES', r'Multi-order Solar EUV Spectrograph'))
    doc.preamble.append(kgpy.latex.Acronym('TRACE', r'Transition Region and Coronal Explorer'))
    doc.preamble.append(kgpy.latex.Acronym('AIA', r'Atmospheric Imaging Assembly'))
    doc.preamble.append(kgpy.latex.Acronym('IRIS', r'Interface Region Imaging Spectrograph'))
    doc.preamble.append(kgpy.latex.Acronym('EIS', 'EUV Imaging Spectrograph'))
    doc.preamble.append(kgpy.latex.Acronym('EUNIS', 'EUV Normal-incidence Spectrometer'))
    doc.preamble.append(kgpy.latex.Acronym('MDI', r'Michelson Doppler Imager'))
    doc.preamble.append(kgpy.latex.Acronym('FUV', 'far ultraviolet'))
    doc.preamble.append(kgpy.latex.Acronym('EUV', 'extreme ultraviolet'))
    doc.preamble.append(kgpy.latex.Acronym('TR', 'transition region'))
    doc.preamble.append(kgpy.latex.Acronym('CTIS', 'computed tomography imaging spectrograph', plural=True))
    doc.preamble.append(kgpy.latex.Acronym('FOV', 'field of view'))
    doc.preamble.append(kgpy.latex.Acronym('SNR', 'signal-to-noise ratio'))
    doc.preamble.append(kgpy.latex.Acronym('PSF', 'point-spread function', plural=True))
    doc.preamble.append(kgpy.latex.Acronym('NRL', 'Naval Research Laboratory'))
    doc.preamble.append(kgpy.latex.Acronym('MHD', 'magnetohydrodynamic'))
    doc.preamble.append(kgpy.latex.Acronym('EE', 'explosive event', plural=True))
    doc.preamble.append(kgpy.latex.Acronym('QS', 'quiet sun'))
    doc.preamble.append(kgpy.latex.Acronym('AR', 'active region'))
    doc.preamble.append(kgpy.latex.Acronym('CH', 'coronal hole'))

    # doc.set_variable(name='ESIS', value=pylatex.Command('ac', 'ESIS'))
    # doc.set_variable(name='MOSES', value=pylatex.Command('ac', 'MOSES'))

    with doc.create(kgpy.latex.Abstract()):
        doc.append(pylatex.NoEscape(
            r"""The \ESIS\ is a next generation rocket borne instrument that will investigate magnetic reconnection 
and energy transport in the solar atmosphere 
\amy{by observing emission lines formed in the chromosphere (\HeI), the transition region (\OV), and corona (\MgX).}
\jake{JDP: Would make more sense to talk about the brighter Mg line?  609.8}
The instrument is a pseudo Gregorian telescope; 
from prime focus, an array of spherical diffraction gratings re-image with differing dispersion angles. 
\amy{The instrument is a pseudo Gregorian telescope with an octagonal field stop at prime focus.  
This field stop is re-imaged  using an array of four spherical diffraction gratings with differing dispersion angles 
relative to ...? [ I want to say relative to solar north or field stop north or something], with each diffraction 
grating projecting the spectrum onto a unique detector.}
The slitless multi-projection design will obtain co-temporal spatial (\avgPlateScale) and spectral (\dispersion) images 
at high cadence ($>=$\minCadence). 
\amy{The instrument is designed to be capable of obtaining co-temporal spatial (\avgPlateScale) and spectral 
(\dispersion) images at high cadence ($>=$\minCadence).}
\amy{Combining the co-temporal exposures from all the detectors will enable us to reconstruct line profile information 
at high spatial and spectral resolution over a large (\fov) \FOV. 
The instrument was launched on September 30, 2019.  The flight data is described in a subsequent paper. }
A single exposure will enable us to reconstruct line profile information at high spatial and spectral resolution over a 
large (\fov) \FOV. 
The instrument is currently in the build up phase prior to spacecraft integration, testing, and launch.
\acresetall"""
        ))

    with doc.create(pylatex.Section('Introduction')):
        doc.append(pylatex.NoEscape(
            r"""The solar atmosphere, as viewed from space in its characteristic short wavelengths (\FUV, \EUV, and soft 
X-ray), is a three-dimensional scene evolving in time: $I[x,y,\lambda,t]$.
Here the solar sky plane spatial coordinates, $x$ and $y$, and the wavelength axis, $\lambda$, comprise the three 
dimensions of the scene, while $t$ represents the temporal axis.
An ideal instrument would capture a spatial/spectral data cube ($I[x,y,\lambda]$) at a rapid temporal cadence ($t$), 
however, practical limitations lead us to accept various compromises of these four observables.
Approaching this ideal is the fast tunable filtergraph (\ie\ fast tunable Fabry--P\'erot etalons, \eg\ the GREGOR 
Fabry--P{\'e}rot Interferometer, \citep{Puschmann12}), but the materials do not exist to extend this technology to 
\EUV\ wavelengths shortward of $\sim$\SI{150}{\nano\meter}~\citep{2000WuelserFP}.
Imagers like the \TRACE~\citep{Handy99} and the \AIA~\citep{Lemen12} collect high cadence 2D \EUV\ spatial scenes, but 
they collect spectrally integrated intensity over a fixed passband that is not narrow enough to isolate a single 
emission line.  
In principle, filter ratios that make use of spectrally adjacent multilayer \EUV\ passbands could detect Doppler 
shifts~\citep{Sakao99}.
However, the passbands of the multilayer coatings are still wide enough that the presence of weaker contaminant lines 
limits resolution of Doppler shifts to $\sim$\SI{1000}{\kilo\meter\per\second}~\citep{Kobayashi00}.
Slit spectrographs (\eg\ the \IRIS~\citep{IRIS14}) obtain fast, high-resolution spatial and spectral observations, but are 
limited by the narrow \FOV\ of the spectrograph slit.
The $I[x,y,\lambda]$ data cube can be built up by rastering the slit pointing, but it cannot be co-temporal along the 
raster axis.
Moreover, extended and dynamic scenes can change significantly in the time required to raster over their extant.  

A different approach is to forego the entrance slit employed by traditional spectrographs entirely.
The \Acposs{NRL} SO82A~\citep{Tousey73,Tousey77} was one of the first instruments to pioneer this method.
The `overlappograms' obtained by SO82A identified several spectral line transitions~\citep{Feldman85}, and have more 
recently been used to determine line ratios in solar flares~\citep{Keenan06}.
Unfortunately, for closely spaced \EUV\ lines, the dispersed images from the single diffraction order suffer from 
considerable overlap confusion.
Image overlap is all but unavoidable with this configuration, however, overlappograms can be disentangled, or inverted, 
under the right circumstances.  

In analogy to a tomographic imaging problem~\citep{Kak88}, inversion of an overlapping spatial/spectral scene can be 
facilitated by increasing the number of independent spectral projections, or `look angles,' through the 3D 
$(x,y,\lambda)$ scene~\citep{DeForest04}.
For example, \citet{DeForest04} demonstrated recovery of Doppler shifts in magnetograms from two dispersed orders of a 
grating at the output of the \MDI~\citep{Scherrer95}.
The quality of the inversion (\eg\ recovery of higher order spectral line moments) can also be improved by additional 
projections~\citep{Kak88,Descour97}, generally at the cost of computational complexity~\citep{Hagen08}.
\Acp{CTIS}~\citep{okamoto1991,Bulygin91,Descour95} leverage this concept by obtaining multiple, simultaneous dispersed 
images of an object or scene; 
upwards of 25 grating diffraction orders may be projected onto a single detector plane~\citep{Descour97}.
Through post processing of these images, \CTIS\ can recover a 3D data cube from a (spectrally) smooth and continuous 
scene over a large bandpass (\eg\ \citet{Hagen08}).

The \MOSES~\citep{Fox10,Fox11} is our first effort aimed at developing the unique capability of simultaneous 
imaging and spectroscopy for solar \EUV\ scenes.
\MOSES\ is a three-order slitless spectrograph that seeks to combine the simplicity of the SO82A concept with the 
advantages of a \CTIS\ instrument.
A single diffraction grating (in conjunction with a fold mirror) projects the $m=\pm1$ and the un-dispersed $m=0$ order 
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

Building on the working concept demonstrated by \MOSES, here we describe a new instrument, the \ESIS, that will improve 
on past efforts to produce a solar \EUV\ spectral map.
\ESIS\ will fly alongside \MOSES\ and will observe the \TR\ and corona of the solar atmosphere in the \OV\ and \MgX\ / 
\MgXdimWavelength\ spectral lines.
In Section~\ref{sec:TheESISConcept} we detail how our experience with the \MOSES\ instrument has shaped the design of 
\ESIS.
Section~\ref{sec:ScienceObjectives} describes the narrow scientific objectives and the requirements placed on the new 
instrument.
Section~\ref{sec:TheESISInstrument} describes the technical approach to meet our scientific objectives, followed by a 
brief description of the expected mission profile in Section~\ref{sec:MissionProfile}.
The current status and progress toward launch is summarized in Section~\ref{sec:ConclusionsandOutlook}."""
        ))

    with doc.create(pylatex.Section('The ESIS Concept')):
        doc.append(pylatex.NoEscape(
            r"""A primary goal of the \ESIS\ instrument is to improve the implementation of \EUV\ snapshot imaging 
spectroscopy demonstrated by \MOSES.  
Therefore, the design of the new instrument draws heavily from experiences and lessons learned through two flights of 
the \MOSES\ instrument.
\ESIS\ and \MOSES\ are both slitless, multi-projection spectrographs.
As such, both produce dispersed images of a narrow portion of the solar spectrum, with the goal of enabling the 
reconstruction of a spectral line profile at every point in the field of view.
The similarities end there, however, as the optical layout of \ESIS\ differs significantly from that of \MOSES.
In this section, we detail some difficulties and limitations encountered with \MOSES, then describe how the new design 
of \ESIS\ addresses these issues."""
        ))
        with doc.create(pylatex.Subsection('Limitations of the MOSES Design')):
            doc.append(pylatex.NoEscape(
                r"""The \MOSES\ design features a single concave diffraction grating forming images on three CCD 
detectors~\citep{Fox10} (Fig.~\ref{fig:mosesSchematic}). 
The optical path is folded in half by a single flat secondary mirror (omitted in Fig.~\ref{fig:mosesSchematic}).
Provided that the three cameras are positioned correctly, this arragement allows the entire telescope to be brought 
into focus using only the central (undispersed) order and a visible light source.
Unfortunately this design uses volume inefficiently for two reasons.
First, the lack of magnification by the secondary mirror limits the folded length of the entire telescope to be no less 
than half the \SI{5}{\meter} focal length of the grating~\citep{Fox10,Fox11}.
Second, the dispersion of the instrument is controlled by the placement of the cameras.
To achieve the maximum dispersion of \SI{29}{\kilo\meter\per\second}~\citep{Fox10}, the outboard orders are imaged as 
far apart as possible in the $\sim22''$ diameter cross section of the rocket payload.
The resulting planar dispersion poorly fills the cylindrical volume of the payload, leaving much unused space along the 
orthogonal planes."""
            ))

            with doc.create(pylatex.Figure(position='!ht')) as moses_schematic:
                moses_schematic.add_image('figures/MOSES_Schematic', width=pylatex.NoEscape(r'\columnwidth'))
                moses_schematic.append(kgpy.latex.Label('fig:mosesSchematic'))
                moses_schematic.add_caption(pylatex.NoEscape(
                    r"""Schematic diagram of the MOSES instrument.
Incident light on the right forms an undispersed image on the central $m=0$ CCD.
Dispersed images are formed on the outboard $m=\pm1$ CCDs."""
                ))
            doc.append(pylatex.NoEscape(
                r"""Furthermore, the monolithic secondary, though it confers the focus advantage noted above, does not 
allow efficient placement of the dispersed image order cameras.  
For all practical purposes, the diameter of the payload (\SI{0.56}{\meter}) can only accommodate three diffraction 
orders ($m=-1, 0, +1$).
Therefore, \textit{\MOSES\ can only collect, at most, three pieces of information at each point in the field of view.}
From this, it is not reasonable to expect the reconstruction of more than three degrees of freedom in this spectrum, 
except in the case very compact, isolated features such as those described by \citet{Fox10} and \citet{Rust17}.
Consequently, it is a reasonable approximation to say that \MOSES\ is sensitive primarily to spectral line intensities, 
shifts, and widths \citep{KankThom01}.
With any tomographic apparatus, the degree of detail that can be resolved in the object depends critically on the 
number of viewing angles~\citep{Kak88,Descour97,Hagen08}.
So it is with the spectrum we observe with \MOSES: more dispersed images are required to confer sensitivity to finer 
spectral details such additional lines in the passband or higher moments of the spectral line shape.

A related issue stems from the use of a single grating, with a single plane of dispersion.
Since the solar corona and transition region are structured by magnetic fields, the scene tends to be dominated by 
field aligned structures such as loops~\citep{Rosner78,Bonnet80}.
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
Since the \MOSES\ grating forms images in three orders simultaneously, aberration cannot be simultaneously optimized for 
all three of those spectral orders.
A result of this design is that the orientations (\ie\,the central axis) of the \PSFs\ vary order to order~\citep{Rust17}.
During the first mission, MOSES was flown with a small amount of defocus~\citep{Rust17}, which exacerbated the 
inter-order \PSF\ variation and caused the individual \PSFs\ to span several pixels~\citep{Rust17,Atwood18}.
The combination of these two effects results in spurious spectral features that require additional 
consideration~\citep{Atwood18} and further increase the complexity of the inversion process~\citep{Rust17,Courrier18}.
 
Another complication is that the spatial and spectral content differs slightly between the three \MOSES\ image orders.
This is because the \MOSES\ \FOV\ is defined by a combination of the aperture of the grating (\ie\,the entrance 
aperture of the telescope) and the spatial extent of the CCDs.
The \FOV\ in the $m=\pm1$ orders is shifted along the dispersion axis as a function of wavelength, dependant upon where 
the dispersed spectral images intercept the $m=\pm1$ CCDs.
Spatially, this effect is limited to only a handful of pixel columns at the edges of each image order.
Of higher concern is the `spectral contamination' allowed by this layout; 
\citet{Parker16} found that bright spectral lines and continuum far outside the wavelength passband and nominal $m=0$ 
\FOV\ could be diffracted onto the outboard order CCDs.
This off-band contamination is detected as systematic intensity variation that lacks an anti-symmetric pairing in the 
opposite dispersed image order.
Analysis of the spectral contamination is ongoing. 

Finally, the exposure cadence of \MOSES\ is hindered by an $\sim$\SI{6}{\second} readout time for the 
CCDs~\citep{Fox11}.
The observing interval for a solar sounding rocket flight is very short, typically about five minutes.
Consequently, every second of observing time is precious, both to achieve adequate exposure time and to catch the full 
development of dynamical phenomena.
The \MOSES\ observing duty cycle is $\sim$\SI{50}{\percent} since it is limited by the readout time of the CCDs.
Thus, valuable observing time is lost.
The readout data gap impelled us to develop a \MOSES\ exposure sequence with exposures ranging from 
$0.25$-\SI{24}{\second}, a careful trade-off between deep and fast exposures. 

In summary, our experience leads us to conclude that the \MOSES\ design has the following primary limitations:
\begin{enumerate}
    \item inefficient use of volume \label{item-length} %(x and y direction)
    \item dispersion constrained by payload dimensions \label{item-disp_con}
    \item too few dispersed images (orders) \label{item-orders}
    \item single plane dispersion \label{item-dispersion}
    \item lack of aberration control \label{item-PSF}
    \item insufficiently defined FOV \label{item-FOV}
    \item sub-optimal exposure cadence \label{item-CAD}
\end{enumerate}
In designing \ESIS, we have sought to improve upon each of these points.
"""
            ))

        with doc.create(pylatex.Subsection('ESIS Features')):
            with doc.create(kgpy.latex.FigureStar(position='!ht')) as esis_figure_3d:
                esis_figure_3d.add_image('figures/layout', width=pylatex.NoEscape(r'\textwidth'))

            with doc.create(kgpy.latex.FigureStar(position='!ht')) as esis_figure_3d:
                esis_figure_3d.add_image(str(figures.layout_pdf()), width=None)
                esis_figure_3d.append(kgpy.latex.Label('fig:layout'))
                esis_figure_3d.add_caption(pylatex.NoEscape(
                    r"""The \ESIS\ instrument is a pseudo-Gregorian design.
The secondary mirror is replaced by a segmented array of concave diffraction gratings.
The field stop at prime focus defines instrument spatial/spectral \FOV.
CCDs are arrayed around the primary mirror, each associated with a particular grating.
Eight grating positions appear in this schematic; only six fit within the volume of the rocket payload.
Four channels are populated for the first flight."""
                ))

            doc.append(pylatex.NoEscape(
                r"""The layout of \ESIS\ (Fig.~\ref{fig:layout}) is a modified form of Gregorian telescope.
Incoming light is brought to focus at an octagonal field stop by a parabolic primary mirror.
In the \ESIS\ layout, the secondary mirror of a typical Gregorian telescope is replaced by a segmented, octagonal array 
of diffraction gratings.
From the field stop, the gratings re-image to CCD detectors arranged radially around the primary mirror.
The gratings are blazed for first order, so that each CCD is fed by a single corresponding grating, and all the 
gratings are identical in design.
The features of this new layout address all of the limitations described in 
Sect.~\ref{subsec:LimitationsoftheMOSESDesign}, and are summarized here.

Replacing the secondary mirror with an array of concave diffraction gratings confers several advantages to \ESIS\ over 
\MOSES.
First, the magnification of the \ESIS\ gratings results in a shorter axial length than MOSES, without sacrificing 
spatial or spectral resolution.
Second, the magnification and tilt of an individual grating controls the position of the dispersed image with respect 
to the optical axis, so that the spectral resolution is not as constrained by the payload dimensions.
Third, the radial symmetry of the design places the cameras closer together, resulting in a more compact instrument.
Furthermore, by arranging the detectors around the optical axis, more dispersed grating orders can be populated; 
up to eight gratings can be arrayed around the \ESIS\ primary mirror (up to six with the current optical table).
This contrasts the three image orders available in the planar symmetry of \MOSES.
Taken together, these three design features make \ESIS\ more compact than \MOSES\ 
(\S\,\ref{subsec:LimitationsoftheMOSESDesign} item~\ref{item-length}), improve spectral resolution 
(\S\,\ref{subsec:LimitationsoftheMOSESDesign} item~\ref{item-disp_con}) and allow the collection of more projections to 
better constrain the interpretation of the data (\S\,\ref{subsec:LimitationsoftheMOSESDesign} item~\ref{item-orders}).
 
The \ESIS\ gratings are arranged in a segmented array, clocked in \SI{45}{\degree} increments, so that there are four 
distinct dispersion planes.
This will greatly aid in reconstructing spectral line profiles since the dispersion space of \ESIS\ occupies a 3D 
volume rather than a 2D plane as with \MOSES.
For \ESIS, there will always be a dispersion plane within \SI{22.5}{\degree} of the normal to any loop-like feature in 
the solar atmosphere.
As discussed in \S\,\ref{subsec:LimitationsoftheMOSESDesign}, a nearly perpendicular dispersion plane allows a 
filamentary structure to serve like a spectrographic slit, resulting in a clear presentation of the spectrum.
This feature addresses \S\,\ref{subsec:LimitationsoftheMOSESDesign} item~\ref{item-dispersion}. 

Rather than forming images at three spectral orders from a single grating, each \ESIS\ imaging channel has a dedicated 
grating.
Aberrations are controlled by optimizing the grating design to form images in first order, over a narrow range of ray 
deviation angles.
This design controls aberration well enough to allow pixel-limited imaging, avoiding the \PSF\ mismatch problems inherent 
to the \MOSES\ design (\S\,\ref{subsec:LimitationsoftheMOSESDesign} item \ref{item-PSF}).
A disadvantage of this arrangement is that \ESIS\ lacks a zero order focus.
In its flight configuration with gratings optimized around a \OVwavelength\ wavelength, the instrument cannot be 
aligned and focused in visible light like \MOSES.
Visible gratings and a special alignment transfer apparatus (\S\,\ref{subsec:AlignmentandFocus}) must be used for 
alignment and focus of \ESIS.

The \ESIS\ design also includes an octagonal field stop placed at prime focus.
This confers two advantages.
First, the field stop fully defines the instrument \FOV, so that \ESIS\ is not susceptible to the spectral contamination 
observed in \MOSES\ data (\S\,\ref{subsec:LimitationsoftheMOSESDesign} limitation~\ref{item-FOV}).
Second, each spectral image observed by \ESIS\ will be bordered by the outline of the field stop 
(\eg\,\S\,\ref{subsec:Optics}).
This aids the inversion process since outside of this sharp edge the intensity is zero for any look angle through an 
ESIS data cube.
Additionally, the symmetry of the field stop gives multiple checkpoints where the edge inversion is duplicated in the 
dispersed images produced by adjacent orders.
The size and octagonal shape of the field stop are defined by the requirement that all CCDs must see the entire \FOV\ 
from edge to edge, while leaving a small margin for alignment. 

Lastly, in contrast to \MOSES, \ESIS\ employs frame transfer CCDs to make optimum use of our five minutes of observing 
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
            ))

    with doc.create(pylatex.Section('Science Objectives')):
        doc.append(pylatex.NoEscape(
            r"""The previous section discussed the qualitative design aspects of \ESIS\ learned from experience with the 
\MOSES\ instrument.  
\MOSES, in turn, demonstrated a working concept of simultaneous \EUV\ imaging and spectroscopy.
This concept adds a unique capability to the science that we can obtain from the \EUV\ solar atmosphere.
\ESIS, sharing the same payload volume as \MOSES, is manifested to fly in 2019.
In this section, we set forth specific scientific objectives for the combined \ESIS/\MOSES mission.
From these objectives, and with an eye toward synergistic operation of \MOSES\ and \ESIS, in 
\S\,\ref{subsec:ScienceRequirements} we derive the quantitative science requirements that drive the \ESIS\ design.

The combined \ESIS/\MOSES/ mission will address the following two overarching science goals: \begin{inparaenum}[(1)] 
\item observe magnetic reconnection in the \TR\label{item-goal1}, and \item map the transfer of energy through the \TR\ 
with emphasis on \MHD\ waves\label{item-goal2}. \end{inparaenum}
These objectives have significant overlap with the missions of \IRIS~\citep{IRIS14}, the \EIS~\citep{Culhane07}
aboard Hinode, the \EUNIS~\citep{Brosius07,Brosius14}, and a long history of FUV and EUV slit spectrographs.
The \ESIS\ instrument, however, can obtain both spatial and spectral information co-temporally.
This will allow us to resolve complicated morphologies of compact \TR\ reconnection events (as was done with 
\MOSES~\citep{Fox11,Rust17,Courrier18}) and observe signatures of \MHD\ waves over a large portion of the solar disk.
Therefore, in support of goal~\ref{item-goal1}, we will use \ESIS\ and \MOSES\ to map flows as a function of time and 
space in multiple \TR\ reconnection events.
To achieve goal~\ref{item-goal2}, we will cross-correlate the evolution at multiple temperatures in the \TR\ to map the 
vertical transport of energy over a wide \FOV.
In the latest configuration, the \MOSES\ optics are optimized around Ne\,\textsc{vii} (\SI{0.5}{\mega\kelvin}).
To achieve our goals, \ESIS\ should have a complementary wavelength choice such that we can observe a reasonably 
isolated emission line formed in the lower \TR."""
        ))

        with doc.create(pylatex.Subsection('Magnetic Reconnection Events')):
            doc.append(pylatex.NoEscape(
                r"""Magnetic reconnection describes the re-arrangement of the magnetic topology wherein magnetic energy 
is converted to kinetic energy resulting in the acceleration of plasma particles.
Reconnection is implicated in many dynamic, high energy solar events.
Solar flares are a well studied example (\eg\,\citet{Priest02} and the references therein), however we have little hope 
of pointing in the right place at the right time to observe a significant flare event in a rocket flight lasting only 
five minutes.
Instead, we will search for signatures of magnetic reconnection in \TR\ spectral lines.      
%label to track table 1 references
\phantomsection
\label{t1_2}
A particular signature of reconnection in the \TR\ is the explosive energy release by ubiquitous, small scale events.
These \EEs\ are characterized as spatially compact ($\approx$\SI{1.5}{\mega\meter} length~\citep{Dere94}) line 
broadenings on the order of \SI{100}{\kilo\meter\per\second}~\citep{Dere91}.
They are observed across a range of \TR\ emission lines that span temperatures of \SI{20000}{}--\SI{250000}{\kelvin} 
(C\,\textsc{ii}--O\,\textsc{v})~\citep{1994Moses}.
The typical lifetime of an \EE\ is 60-\SI{90}{\second}~\citep{1994Moses,Dere94,Dere91}.
Due to their location near quiet sun magnetic network elements, and the presence of supersonic flows near the Alfv\`en 
speed, \citet{Dere91} first suggested that \EEs\ may result from the model of fast Petschek~\citep{Petschek64} 
reconnection. 

The spectral line profile of \EEs\ may indicate the type of reconnection that is occurring in the \TR\ 
(\eg\,\citet{Rust17}).
For example, the Petschek model of reconnection predicts a `bi-directional jet' line profile with highly Doppler 
shifted wings, but little emission from the line core~\citep{Innes99}.
\citet{Innes15} developed a reconnection model resulting from a plasmoid instability~\citep{Bhattacharjee09}.
In contrast to the bi-directional jet, this modeled line profile has bright core emission and broad wings.
Both types of profile are seen in slit spectrograph data (\eg, \citet{Innes97,Innes15}, and the references therein), 
however \MOSES\ observed \EEs\ with more complicated morphologies than either of these two models 
suggest~\citep{Fox10,Rust17}.
It is unclear whether the differing observations are a function of wavelength and temperature, a result of a limited 
number of observations, or because the morphology of the event is difficult to ascertain from slit spectrograph data.

%label to track table 1 references
\phantomsection
\label{t1_01}
\ESIS\ will observe magnetic reconnection in the context of \EEs, by extending the technique pioneered by \MOSES\ to 
additional \TR\ lines.
Explosive events are well suited to sounding rocket observations;
a significant portion of their temporal evolution can be captured in $>$\SI{150}{\second} (\eg\,the analysis by 
\citet{Rust17}) and they are sufficiently common to provide a statistically meaningful sample in a 5-minute rocket 
flight (\eg,~\citet{Dere89,Dere91}).
In similarity with \MOSES, we seek a \TR\ line for \ESIS\ that is bright and well enough isolated from neighboring 
emission lines so as to be easily distinguished."""
            ))

        with doc.create(pylatex.Subsection('Energy Transfer')):
            doc.append(pylatex.NoEscape(
                r"""Tracking the mass and energy flow through the solar atmosphere is a long-standing goal in solar 
physics.
Bulk mass flow is evidenced by Doppler shifts or skewness in spectral lines.
However, the observed non-thermal broadening of \TR\ spectral lines may result from a variety of physical processes, 
including \MHD\ waves~\citep{DePontieu15, DePontieu07}, high-speed evaporative up-flows (\eg\,nanoflares, 
\citet{Patsourakos06}), turbulence, and other sources (\eg\,\citet{Mariska1992}).
This is a broad topic which \ESIS\ can address in many ways.
Here we will focus on a single application;
\ESIS\ will search for sources of Alfv\'en waves in the solar atmosphere by observing line broadening as the 
spectroscopic signature of these waves.

Alfv\'en waves in coronal holes are observed to carry an energy flux of 
\SI{7e5}{erg\per\centi\square\meter\per\second}, enough to energize the fast solar wind \citep{Hahn2012,Hahn2013}.
The source and frequency spectrum of these waves is unknown.
Here, we hypothesize that \MHD\ waves are similarly ubiquitous in quiet Sun and active regions, and play an important 
role in the energization of the quiescent corona.

%label to track table 1 references
\phantomsection
\label{t1_1}
The magnitude of non-thermal broadening of optically thin spectral lines is a direct measure of the wave 
amplitude~\citep{Banerjee09,Hahn2012,Hahn2013}.
We may estimate a lower limit on the non-thermal velocity to be observed as follows.
We assume that the magnetic field is constant for small changes in scale height in the \TR\ and that line of sight 
effects are negligible for observations sufficiently far from disk center.
Since the solar wind is not accelerated to an appreciable fraction of the Alfv\'en wave velocity at altitudes below 
$R \leq 1.15R_\odot$~\citep{Cranmer05}, the wave amplitude, $v_{nt}$, depends only weakly on electron density, $n_e$, so 
that $v_{nt} \propto n_e^{-1/4}$~\citep{Hahn2013,Moran01}.
Assuming pressure balance between the low corona and transition zone, we may infer non-thermal velocities in the \TR\ by 
scaling according to the temperature drop, $v_{nt} \propto T^{1/4}$.
The measured non-thermal velocity of \SI{24}{\kilo\meter\per\second} for Si\,\textsc{viii}~\citep{Doyle98} 
(\SI{0.8}{\mega\kelvin}~\citep{Moran03}) near the limb should, neglecting damping, correspond to velocities of at least 
\SI{21}{\kilo\meter\per\second} in mid \TR\ Ne\,\textsc{vii}, and \SI{18}{\kilo\meter\per\second} in the lower 
O\,\textsc{v} (\SI{0.25}{\mega\kelvin}) line.
The above non-thermal velocities are arrived at assuming both O\,\textsc{v} and Ne\,\textsc{vii} are formed near their 
ionization equilibrium temperatures.
For O\,\textsc{v}, the thermal width is $\sim$\SI{11}{\kilo\meter\per\s} at \SI{0.25}{\mega\kelvin} which means the 
total linewidth is primarily due to the non-thermal component.    
    
More recently, ~\citet{Srivastava17} observed torsional Alfv\'en waves with amplitude 
$\sim$\SI{20}{\kilo\meter\per\second} and period $\sim$\SI{30}{\second} in the chromosphere.
Modeling shows that these torsional waves can transfer a significant amount of energy to the corona~\citep{Kudoh99}.
The torsional motion will be observed as Doppler shifts when viewed from the side.
The oscillation period is long enough to be well resolved but short enough to see $\sim$\SI{10}{} cycles in a single 
rocket flight.
An \ESIS-like instrument is therefore well suited to observations of torsional Alfv\'en wave propagation over multiple 
heights in the \TR. 

By mapping Doppler velocities over a wide field of view in the \TR, \ESIS\ can address questions about both the origin 
of waves and whether they are able to propagate upward into the corona.
Independent of the two propagation modes discussed above, there is a range of possible sources for Alfv\'en 
(and other \MHD) waves in the solar atmosphere.
Three potential scenarios are: \begin{inparaenum}[(1)] \item Waves originate in the chromosphere or below and propagate 
through the \TR\ at a spatially uniform intensity; \label{wave-1}
\item Intense sources are localized in the \TR, but fill only a fraction of the surface\label{wave-2}; and \item Weak 
sources are localized in the \TR, but cover the surface densely enough to appear like the first case\label{wave-3}. 
\end{inparaenum}
The resulting non-thermal widths for localized sources will be significantly higher than the 
$\sim$\SI{20}{\kilo\meter\per\second} mean derived above.
The concentration of non-thermal energy observed by \ESIS\ will serve as an indicator of source density.
\roy{Should we remove this section considering that \MOSES\ didn't work?}
Comparison of Doppler maps from \ESIS\ and \MOSES\ will indicate whether a uniform source density originates in the 
chromosphere or below (scenario~\ref{wave-1}) or is associated with spatially distributed \TR\ phenomena 
(scenario~\ref{wave-3}) such as explosive events, or macrospicules.
Comparison with a wider selection of ground and space based imagery will allow us to determine whether intense, 
localized sources (scenario~\ref{wave-2}) are associated with converging or emerging magnetic bipoles, type \textsc{ii} 
spicules, spicule bushes, or other sources beneath the \TR.
For these comparisons, we need only to localize, rather than resolve, wave sources.
A spatial resolution of $\sim$\SI{2}{\mega\meter} will be sufficient to localize sources associated with magnetic flux 
tubes that are rooted in photospheric inter-granular network lanes (\eg\,\citet{Berger95ApJ})."""
            ))

            with doc.create(pylatex.Subsection('Science Requirements')):
                doc.append(pylatex.NoEscape(
                    r"""\ESIS\ will investigate two science targets; 
reconnection in explosive events, and the transport of mass and energy through the transition region.
The latter may take many forms, from \MHD\ waves of various modes to \EUV\ jets or macro-spicules.
To fulfill these goals, \ESIS\ will obtain simultaneous intensity, Doppler shift and line width images of the \OV\ line 
in the solar transition region at rapid cadence.
This is a lower \TR\ line (\SI{.25}{\mega\kelvin}) that complements the MOSES Ne\,\textsc{vii}.
The bright, optically thin \OVion\ emission line is well isolated except for the two coronal \MgXion\ lines.
These coronal lines can be viewed as contamination or as a bonus;
we expect that with the four \ESIS\ projections it will be possible to separate the \OVion\ emission from that of 
\MgXion.
From the important temporal, spatial, and velocity scales referenced Sections~\ref{subsec:MagneticReconnectionEvents} 
and \ref{subsec:EnergyTransfer} we define the instrument requirements in Table~\ref{table:scireq} that are needed to 
meet our science goals."""
                ))

                with doc.create(pylatex.Table()) as table:
                    table._star_latex_name = True
                    table.append(kgpy.latex.Label('table:scireq'))
                    with table.create(pylatex.Center()) as centering:
                        with centering.create(pylatex.Tabular(table_spec='llll', )) as tabular:
                            tabular.escape = False
                            tabular.add_row(['Parameter', 'Requirement', 'Science Driver', 'Capabilities'])
                            tabular.add_hline()
                            tabular.add_row([
                                r'Spectral line',
                                r'\OVion\ \roy{\OV}',
                                r'Explosive events',
                                r'\OVion, \MgXion, \HeIion, Table~\ref{table:prescription}',
                            ])
                            tabular.add_row([
                                r'Spectral resolution',
                                r'\SI{18}{\kilo\meter\per\second} broadening',
                                r'\MHD\ waves',
                                r'\dispersionDoppler, Table~\ref{table:prescription}',
                            ])
                            tabular.add_row([
                                r'Spatial resolution',
                                r'\SI{2}{\arcsecond} (\SI{1.5}{\mega\meter})',
                                r'Explosive events',
                                r'\minSpatialResolution, Table~\ref{table:prescription}',
                            ])
                            tabular.add_row([
                                r'Desired \SNR',
                                r'\SI{17.3}{} in \CH',
                                r'\MHD\ waves in \CH',
                                r'$>$\SI{17.7}{} w/$20\times$\SI{10}{\second} exp., '
                                r'\S~\ref{subsec:SensitivityandCadence}',
                            ])
                            tabular.add_row([
                                r'Cadence',
                                r'\SI{15}{\second}',
                                r'Torsional waves',
                                r'\SI{10}{\second} eff., \S~\ref{subsec:SensitivityandCadence}',
                            ])
                            tabular.add_row([
                                r'Observing time',
                                r'$>$\SI{150}{\second}',
                                r'Explosive events',
                                r'\SI{270}{\second}, \S~\ref{sec:MissionProfile}',
                            ])
                            tabular.add_row([
                                r'\FOV',
                                r'\SI{10}{\arcminute} diameter',
                                r'Span \QS, \AR, and limb',
                                r'\fov, Table~\ref{table:prescription}',
                            ])
                        table.add_caption(pylatex.NoEscape(
                            r"""ESIS instrument requirements.  
AR is active region, QS quiet sun, and CH coronal hole."""
                        ))

    with doc.create(pylatex.Section('The ESIS Instrument')):

        doc.append(pylatex.NoEscape(
            r"""\ESIS\ is a multiple projection slitless spectrograph that obtains line intensities, Doppler shifts, and 
widths in a single snapshot over a 2D \FOV.
Starting from the notional instrument described in Sec.~\ref{sec:TheESISConcept}, \ESIS\ has been designed to ensure all 
of the science requirements set forth in Table~\ref{table:scireq} are met.
The final design parameters are summarized in Table~\ref{table:prescription}.

A schematic diagram of a single \ESIS\ channel is presented in Fig.~\ref{F-ESIS_AP} [A], while the mechanical features 
of the primary mirror and gratings are detailed in Figs.~\ref{F-ESIS_AP} [B] and [C], respectively."""
        ))

        with doc.create(pylatex.Table()) as table:
            table._star_latex_name = True
            table.append(kgpy.latex.Label('table:prescription'))
            with table.create(pylatex.Center()) as centering:
                with centering.create(pylatex.Tabular('ll')) as tabular:
                    tabular.escape = False
                    tabular.add_hline()
                    tabular.add_row(['Primary', 'Parabolic'])
                    tabular.add_row(['', r'Octagonal aperture, D=\primaryDiameter'])
                    tabular.add_row(['', r'\roy{Octagonal aperture, \primaryDiameter\ diameter}'])

        with doc.create(pylatex.Subsection('Optics')):
            pass

        with doc.create(pylatex.Subsection('Optimization and Tolerancing')):
            pass

        with doc.create(pylatex.Subsection('Coatings and Filters')):
            pass

        with doc.create(pylatex.Subsection('Sensitivity and Cadence')):
            pass

        with doc.create(pylatex.Subsection('Alignment and Focus')):
            pass

        with doc.create(pylatex.Subsection('Apertures and Baffles')):
            pass

        with doc.create(pylatex.Subsection('Cameras')):
            pass

        with doc.create(pylatex.Subsection('Avionics')):
            pass

        with doc.create(pylatex.Subsection('Pointing System')):
            pass

        with doc.create(pylatex.Subsection('Mechanical')):
            pass

    with doc.create(pylatex.Section('Mission Profile')):
        pass

    with doc.create(pylatex.Section('Conclusions and Outlook')):
        pass

    doc.append(pylatex.Command('bibliography', arguments='sources'))

    return doc


if __name__ == '__main__':
    # plt.rcParams['axes.labelsize'] = 9
    # plt.rcParams['xtick.labelsize'] = 9
    # plt.rcParams['ytick.labelsize'] = 9
    # plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['font.size'] = 9
    plt.rcParams['lines.linewidth'] = 1

    doc = document()
    doc.generate_pdf()
    doc.generate_tex()
