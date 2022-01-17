import typing as typ
import pathlib
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.modeling
import astropy.visualization
import numpy as np
import pylatex
import kgpy.format
import kgpy.latex
import kgpy.units
import kgpy.chianti
import kgpy.optics
import esis.optics
import esis.science.papers.instrument.figures as figures
from . import optics
from . import preamble
from . import variables
from . import authors
from . import tables
from . import sections

path_base = pathlib.Path(__file__).parent
path_pdf = path_base / 'instrument'
path_figures = path_base / 'figures'


def document() -> kgpy.latex.Document:

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 9
    plt.rcParams['lines.linewidth'] = 1

    doc = kgpy.latex.Document(
        default_filepath=str(path_pdf),
        documentclass='aastex631',
        document_options=[
            'twocolumn',
            # 'linenumbers',
        ]
    )

    doc.packages.append(pylatex.Package('paralist'))
    doc.packages.append(pylatex.Package('amsmath'))
    doc.packages.append(pylatex.Package('acronym'))
    doc.packages.append(pylatex.Package('savesym'))

    doc.preamble += preamble.body()

    variables.append_to_document(doc)

    doc.append(kgpy.latex.Title('The EUV Snapshot Imaging Spectrograph'))

    doc += authors.author_list()

    optics_single = optics.as_designed_single_channel()
    optics_all = esis.flight.optics.as_measured()
    wavelength = optics_single.bunch.wavelength
    index_o5 = np.nonzero(optics_single.bunch.ion == 'o_5')[0][0]
    wavelength_o5 = wavelength[index_o5]
    index_mg10_2 = np.nonzero(optics_single.bunch.ion == 'mg_10')[0][1]
    wavelength_mg10_2 = wavelength[index_mg10_2]

    doc.append(sections.abstract.section())

    doc.append(sections.introduction.section())

    doc.append(sections.esis_concept.section())

    doc.append(sections.science_objectives.section())

    with doc.create(pylatex.Section(pylatex.NoEscape('The \ESIS\ Instrument'))):
        doc.append(pylatex.NoEscape(
            r"""\ESIS\ is a multi-projection slitless spectrograph that obtains line intensities, Doppler shifts, and 
widths in a single snapshot over a 2D \FOV.
Starting from the notional instrument described in Sec.~\ref{sec:TheESISConcept}, \ESIS\ has been designed to ensure all 
of the science requirements set forth in Table~\ref{table:scireq} are met.
The final design parameters are summarized in Table~\ref{table:prescription}.

A schematic diagram of a single \ESIS\ channel is presented in Fig.~\ref{fig:schematic}a, while the mechanical features 
of the primary mirror and gratings are detailed in Figs.~\ref{fig:schematic}b and \ref{fig:schematic}c, respectively."""
        ))

        doc.append(tables.prescription.table())

        doc.append(sections.esis_instrument.optics.subsection())

        doc.append(sections.esis_instrument.optimization_and_tolerancing.subsection(doc))

        doc.append(sections.esis_instrument.vignetting.subsection())

        doc.append(sections.esis_instrument.distortion.subsection())

        doc.append(sections.esis_instrument.coatings_and_filters.subsection())

        doc.append(sections.esis_instrument.sensitivity_and_cadence.subsection(doc))

        doc.append(sections.esis_instrument.alignment_and_focus.subsection())

        doc.append(sections.esis_instrument.apertures_and_baffles.subsection())

        doc.append(sections.esis_instrument.cameras.subsection())

        doc.append(sections.esis_instrument.avionics.subsection())

        with doc.create(pylatex.Subsection('Pointing System')):
            doc.append(pylatex.NoEscape(
                r"""
The imaging target will be selected prior to launch, the morning of the day of flight.
During flight, pointing will be maintained by the \SPARCS\ \citep{Lockheed69}.
Images from Camera 1 will be downlinked and displayed in real time on the \SPARCS\ control system console at intervals of 
$\sim$\SI{16}{\second} to verify pointing is maintained during flight."""
            ))

        with doc.create(pylatex.Subsection('Mechanical')):
            doc.append(pylatex.NoEscape(
                r"""
\ESIS\ and \MOSES\ are mounted on opposite sides of a composite optical table structure originally developed for the 
\SPDE~\citep{Bruner95lock}.
The layered carbon fiber structure features a convenient, precisely coplanar array of threaded inserts with precision 
counterbores.
The carbon fiber layup is designed to minimize the longitudinal coefficient of thermal expansion.
The optical table is housed in two \SI{0.56}{\meter} diameter skin sections, with a total length of \SI{3}{\meter}.
A ball joint and spindle assembly on one end and flexible metal aperture plate on the other hold the optical table in 
position inside the skin sections. 
The kinematic mounting system isolates the optical table from bending or twisting strain of the skins."""
            ))

    with doc.create(pylatex.Section('Mission Profile')):

        doc.append(pylatex.NoEscape(
            r"""
\ESIS\ will be launched aboard a sub-orbital Terrier Black Brant sounding rocket from White Sands Missile Range.
The experiment is currently scheduled for launch in August, 2019.
Trajectory will follow a roughly parabolic path, with $>$\SI{270}{\second} solar observing time above 
\SI{160}{\kilo\meter}.
\ESIS\ will begin continuously taking exposures at a fixed cadence immediately after launch, terminating just before the 
payload impacts the upper atmosphere.
Exposure length will be determined by the target selected for launch day.
Exposures taken while the payload shutter door is closed ($<$ \SI{160}{\kilo\meter}) will be used for dark calibration.
Data will be stored on board and downloaded after recovery, however a limited amount of data will be transmitted to the 
ground station via high speed telemetry as a safeguard against payload loss or destruction.
A parachute will slow the descent of the payload after it enters the atmosphere, and recovery will be accomplished by 
helicopter after the payload is located on the ground."""
        ))

        with doc.create(pylatex.Subsection(pylatex.NoEscape('\ESIS\ Mission Update'))):
            doc.append(pylatex.NoEscape(
                r"""
Since the time of writing \ESIS\ launched and was recovered successfully from White Sands Missile Range on 
September 30, 2019.
Unfortunately, due to failure of the mechanical shutter, no \MOSES\ data was obtained during this flight.
A  paper is forthcoming that will document the \ESIS\ instrument in its as-flown configuration~\citep{Courrier_inprep}.
A companion paper will describe \ESIS\ first results~\citep{Parker_inprep}.
Two significant changes, one to the \ESIS\ instrument and one to our alignment procedures, were made prior to launch and 
are summarized below.

The transfer from visible to \EUV\ grating alignment was completed by an alternative means.
The apparatus described by~\citet{Johnson18} was not able to maintain sufficient repeatability during test runs on 
diffraction grating surfaces.
To maintain the launch schedule, a phase shifting interferometer was used to transfer the alignment of the visible 
gratings to the \EUV\ flight gratings. 

A trade study was conducted, and it was decided to remove the primary aperture stop. The advantage was an increase in 
sensitivity.
The disadvantage was to sacrifice the unvignetted design described in Section \ref{subsec:AperturesandBaffles}.
The effective aperture is increased by a factor of 1.7 to 2.7 as a function of \FOV\ in the radial dimension.
The corresponding signal gradient is oriented along the dispersion direction of each channel;
vignetting increases (and signal decreases) when moving towards blue wavelengths 
(\ie\,moving to the left in Figure~\ref{fig:projections}).
This gradient is due almost entirely to vignetting by the central obscuration, and is linear across the entire \FOV.
The principal challenge is that the images cannot be corrected directly;
rather, since the gradient is repeated for each of the overlapping spectral line images, the vignetting can only be 
accounted for by forward modeling.
Since forward modeling is required for all of the inversion procedures under consideration for \ESIS\ data analysis, the 
vignetting was deemed low risk to the mission science."""
            ))

    with doc.create(pylatex.Section('Conclusions and Outlook')):
        doc.append(pylatex.NoEscape(
            r"""
\ESIS\ is a next generation slitless spectrograph, designed to obtain co-temporal spectral and spatial images of the solar 
transition region and corona.
In this report, we present details of the scientific objectives, instrument, image and spectral resolution, data 
acquisition, and flight profile.

\ESIS\ follows on the proven \MOSES\ design, incorporating several design changes to improve the utility of the instrument.
The symmetrical arrangement of \CCDs\ and diffraction gratings results in a compact instrument while increasing the number 
of dispersed images and dispersion planes.
This aids the inversion process, while also allowing access to higher order spectral line profile moments.
Individual gratings improve resolution by controlling aberration in each channel.
The addition of a field stop eliminates spectral contamination and provides an easily recognizable edge for data 
inversion.
The \ESIS\ design also demonstrates that all this can be accomplished in a volume small enough to serve as a prototype for 
a future orbital instrument.

For the first flight, four of the six available \ESIS\ channels will be populated with optics optimized around the 
O\,\textsc{v} emission line.
The large (\SI{11.3}{\arcminute} \roy{\fov}), high resolution \FOV\ (\SI{1.52}{\arcsecond} \roy{\spatialResolutionMax}, \SI{74}{\milli\angstrom} \roy{\spectralResolution}) can 
simultaneously observe the evolution of small scale \EUV\ flows and large scale \MHD\ waves in high temporal cadence. 
\ESIS\ also enables the study of transport of mass and energy in the transition region and corona during the $\sim 5$ 
minute data collection portion of rocket flight.

\ESIS\ was recovered after a successful first launch on September 30, 2019, with analysis of collected data currently 
in-process.
Subsequent flights will be proposed and the instrument refined with an eye toward orbital opportunities.
Suborbital flights will allow us to expand the instrument to its full complement of six channels and refine our data 
analysis methods, but do not provide access to major flares and eruptive events that drive space weather.
The long term prospect is that an \ESIS-like instrument on an orbital platform could provide high cadence maps of 
spectral line profiles in solar flares, allowing unique and comprehensive observations of the dynamics in solar eruptive 
events, flare ribbons, and the flare reconnection region."""
        ))

    doc.append(pylatex.Command('bibliography', arguments='sources'))

    return doc
