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
        documentclass='aastex63',
        document_options=[
            'twocolumn',
            # 'linenumbers',
        ]
    )

    doc.packages.append(pylatex.Package('acronym'))
    doc.packages.append(pylatex.Package('savesym'))
    doc.preamble.append(pylatex.NoEscape(
        '\\savesymbol{tablenum}'
        '\\usepackage{siunitx}'
        '\\restoresymbol{SIX}{tablenum}'
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

    doc.preamble.append(pylatex.Command('bibliographystyle', 'aasjournal'))

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
        name='dispersion',
        value=optics_single.dispersion.to(kgpy.units.mAA / u.pix),
        digits_after_decimal=0,
    )

    doc.set_variable_quantity(
        name='minCadence',
        value=optics_single.detector.exposure_length_min,
        digits_after_decimal=1,
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
    doc.preamble.append(kgpy.latex.Acronym('MDI', r'Michelson Doppler Imager'))
    doc.preamble.append(kgpy.latex.Acronym('FUV', 'far ultraviolet'))
    doc.preamble.append(kgpy.latex.Acronym('EUV', 'extreme ultraviolet'))
    doc.preamble.append(kgpy.latex.Acronym('TR', 'transition region'))
    doc.preamble.append(kgpy.latex.Acronym('CTIS', 'computed tomography imaging spectrograph', plural=True))
    doc.preamble.append(kgpy.latex.Acronym('FOV', 'field of view'))
    doc.preamble.append(kgpy.latex.Acronym('NRL', 'Naval Research Laboratory'))

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
spectroscopy demonstrated by \MOSES\.  
Therefore, the design of the new instrument draws heavily from experiences and lessons learned through two flights of 
the \MOSES\ instrument.
\ESIS\ and \MOSES\ are both slitless, multi-projection spectrographs.
As such, both produce dispersed images of a narrow portion of the solar spectrum, with the goal of enabling the 
reconstruction of a spectral line profile at every point in the field of view.
The similarities end there, however, as the optical layout of \ESIS\ differs significantly from that of \MOSES\.
In this section, we detail some difficulties and limitations encountered with \MOSES\, then describe how the new design 
of \ESIS\ addresses these issues."""
        ))
        with doc.create(pylatex.Subsection('Limitations of the MOSES Design')):
            pass

        with doc.create(pylatex.Subsection('ESIS Features')):
            with doc.create(kgpy.latex.FigureStar(position='!ht')) as esis_figure_3d:
                esis_figure_3d.add_image('figures/layout', width=pylatex.NoEscape(r'\textwidth'))

            with doc.create(kgpy.latex.FigureStar(position='!ht')) as esis_figure_3d:
                esis_figure_3d.add_image(str(figures.layout_pdf()), width=None)

    with doc.create(pylatex.Section('Science Objectives')):
        pass

    with doc.create(pylatex.Section('The ESIS Instrument')):
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
