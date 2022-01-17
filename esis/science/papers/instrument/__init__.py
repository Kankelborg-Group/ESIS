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

        with doc.create(pylatex.Subsection('Cameras')):

            doc.append(figures.cameras.figure())

            doc.append(pylatex.NoEscape(
                r"""
The \ESIS\ \CCD\ cameras were designed and constructed by \MSFC\ and are the latest in a 
series of camera systems developed specifically for use on solar space flight instruments.
The \ESIS\ camera heritage includes those flown on both the \CLASP~\citep{Kano12,Kobayashi12} and \HiC~\citep{Kobayashi2014}.

The \ESIS\ detectors are CCD230-42 \roy{\detectorName} astro-process \CCDs\ from E2V \roy{\detectorManufacturer}.
For each camera, the \CCD\ is operated in a split frame transfer mode with each of the four ports read out by a 16-bit \roy{\detectorAnalogToDigitalBits-bit} A/D 
converter.
The central $2048 \times 1024$ \roy{$\detectorPixelsX \times \detectorPixelsY$} pixels of the $2k\times2k$ device are used for imaging, while the outer two regions are 
used for storage.
Two \roy{\DetectorNumOverscanColumnWords} overscan columns on either side of the imaging area and eight extra rows in each storage region will monitor read 
noise and dark current.
When the camera receives the trigger signal, it transfers the image from the imaging region to the storage regions and 
starts image readout.
The digitized data are sent to the \DACS\ through a SpaceWire interface immediately, 
one line at a time.
The frame transfer takes $<$\SI{60}{\milli\second} \roy{\detectorFrameTransferTime}, and readout takes \SI{1.1}{\second} \roy{\detectorReadoutTime}.
The cadence is adjustable from 2-\SI{600}{\second} \roy{\detectorExposureLengthRange} in increments of \SI{100}{\milli\second} \roy{\detectorExposureLengthIncrement}, to satisfy the requirement 
listed in Table~\ref{table:scireq}.
Because the imaging region is continuously illuminated, the action of frame transfer (transferring the image from the 
imaging region to the storage regions) also starts the next exposure without delay.
Thus the exposure time is controlled by the time period between triggers.
Camera 1 \roy{\detectorTriggerIndex} (Fig.~\ref{F-cameras}) generates the sync trigger, which is fed back into Camera 1's \roy{\detectorTriggerIndex's} trigger input and provides 
independently buffered triggers to the remaining three cameras.
The trigger signals are synchronized to better than $\pm$\SI{1}{\milli\second} \roy{$\pm$\detectorSynchronizationError}.
Shutterless operation allows \ESIS\ to observe with a \SI{100}{\percent} duty cycle.
The cadence is limited only by the 1.1\,s \roy{\detectorReadoutTime} readout time. 

\MSFC\ custom designed the camera board, enclosure, and mounting structure for \ESIS\ to fit the unique packaging 
requirements of this experiment (Fig~\ref{F-cameras}).
The front part of the camera is a metal block which equalizes the temperature across the \CCD\ while fastening it in 
place.
The carriers of all cameras are connected to a central two-piece copper (\SI{3}{\kilo\gram}) and aluminum 
(\SI{1}{\kilo\gram}) thermal reservoir (cold block) by flexible copper cold straps.
The flexible cold straps allow individual cameras to be translated parallel to the optical axis (by means of shims) up 
to $\sim$\SI{13}{\milli\meter} \roy{$\sim$\detectorFocusAdjustmentRange} to adjust focus in each channel prior to launch.
The centrally located cold block will be cooled by LN2 \roy{\LN} flow from outside the payload until just before launch.
The LN2 \roy{\LN} flow will be controlled automatically by a Ground Support Equipment (GSE) \roy{\GSE} computer so that all cameras are 
maintained above survival temperature but below the target temperature of \SI{-55}{\celsius} \roy{\detectorTemperatureTarget} to insure a negligible dark 
current level.

The gain, read noise, and dark current of the four cameras were measured at \MSFC\ using an ${}^{55}$Fe radioactive 
source.
Cameras are labeled 1, 2, 3, and 4 \roy{\channelNames} with associated serial numbers SN6, SN7, SN9, and SN10 \roy{\detectorSerialNumbers} respectively in 
Fig.~\ref{F-cameras}.  Gain ranges from 2.5-\SI{2.6}{e^- \per DN} \roy{\detectorGainRange} in each quadrant of all four cameras.
Table~\ref{T-cameras} lists gain, read noise, and dark current by quadrant for each camera.  

The \QE\ of the \ESIS\ \CCDs\ will not be measured before flight.
Similar astro-process \CCDs\ with no AR \roy{antireflection (because AR is already used for active region)} coating are used in the \SXI\ aboard the \GOES\ N and O.
A \QE\ range of 43\% at 583\AA\ \roy{\detectorQuantumEfficiencyHeI\ at \HeI} to 33\% at 630\AA\ \roy{\detectorQuantumEfficiency\ at \OV} is expected for the \ESIS\ \CCDs, based on \QE\ measurements by 
\citet{Stern04} for \GOES\ \SXI\ instruments.

\begin{table}[!htb]
\caption{\ESIS\ Camera properties.}
%\tableformat
\begin{tabular}{ccccc}
Camera & Quad & Gain & Read Noise & Dark Current \\
 & & [$e^-/DN$] & [DN] & [$e^-/ms$] \\
\hline %-----------------------------------------------------------------------------
1 (SN6) & 1 & 2.57 & 3.9 & $1.37e^-4$ \\
& 2 & 2.50 & 4.0 & $9.66e^-5$ \\
& 3 & 2.52 & 4.1 & $6.85e^-5$ \\
& 4 & 2.53 & 3.7 & $9.80e^-5$ \\ 
\hline
2 (SN7) & 1 & 2.55 & 3.9 & $6.77e^-5$ \\
& 2 & 2.58 & 4.0 & $5.89e^-5$ \\
& 3 & 2.57 & 4.0 & $8.98e^-5$ \\
& 4 & 2.63 & 4.0 & $1.01e^-4$ \\ 
\hline %-----------------------------------------------------------------------------
3 (SN9) & 1 & 2.57 & 4.1 & $3.14e^-5$ \\
& 2 & 2.53 & 4.1 & $2.68e^-5$ \\
& 3 & 2.52 & 4.1 & $3.18e^-5$ \\
& 4 & 2.59 & 4.3 & $3.72e^-5$ \\ 
\hline
4 (SN10) & 1 & 2.60 & 3.9 & $6.39e^-4$ \\
& 2 & 2.60 & 3.9 & $5.07e^-5$ \\
& 3 & 2.54 & 4.2 & $6.63e^-5$ \\
& 4 & 2.58 & 4.1 & $8.24e^-5$ \\ 
\hline
\end{tabular}
\label{T-cameras}
\end{table}"""
            ))

            detector = optics_all.detector

            with doc.create(pylatex.Table()) as table:
                # table._star_latex_name = True
                with table.create(pylatex.Center()) as centering:
                    with centering.create(pylatex.Tabular('ccccc')) as tabular:
                        tabular.escape = False
                        tabular.add_row([r'Channel', r'Quad.', r'Gain', r'Read noise', r'Dark current',])
                        tabular.add_row(['', '', f'({detector.gain.unit:latex_inline})', f'({detector.readout_noise.unit:latex_inline})', f'({detector.dark_current.unit:latex_inline})'])
                        tabular.add_hline()
                        for i in range(detector.gain.shape[0]):
                            for j in range(detector.gain.shape[1]):
                                if j == 0:
                                    channel_name_i = optics_all.channel_name[i]
                                    serial_number_i = f'({detector.serial_number[i]})'
                                else:
                                    channel_name_i = ''
                                    serial_number_i = ''
                                tabular.add_row([
                                    f'{channel_name_i} {serial_number_i}',
                                    j + 1,
                                    detector.gain[i, j].value,
                                    detector.readout_noise[i, j].value,
                                    f'{detector.dark_current[i, j].value:0.3f}',
                                ])
                            tabular.add_hline()
                table.add_caption(pylatex.NoEscape(r'\ESIS\ camera properties'))
                table.append(kgpy.latex.Label('tabel:cameraProperties'))

        with doc.create(pylatex.Subsection('Avionics')):
            doc.append(pylatex.NoEscape(
                r"""
The \ESIS\ \DACS\ are based on the designs used for both \CLASP~\citep{Kano12,Kobayashi12} and \HiC~\citep{Kobayashi2014}.
The electronics are a combination of \MOTS\ hardware and custom designed components.
The \DACS\ is a 6-slot, 3U, open VPX PCIe architecture conduction cooled system using an AiTech C873 single board
computer.
The data system also include a \MOTS\ PCIe switch card, \MSFC\ parallel interface card, and two \MOTS\ Spacewire cards.
A slot for an additional Spacewire card is included to accommodate two more cameras for the next \ESIS\ flight.
The C873 has a \SI{2.4}{\giga\hertz} Intel i7 processor with \SI{16}{\giga b} of memory.
The operating temperature range for the data system is -40 to +85 C.
The operating system for the flight data system is Linux Fedora 23.

The \DACS\ is responsible for several functions;
it controls the \ESIS\ experiment, responds to timers and uplinks, acquires and stores image data from the cameras, 
downlinks a subset of images through telemetry, and provides experiment health and status.
The \DACS\ is housed with the rest of the avionics (power supply, analog signal conditioning system) in a 
0.56-\SI{0.43}{\meter} transition section outside of the experiment section.
This relaxes the thermal and cleanliness constraints placed on the avionics.
Custom DC/DC converters are used for secondary voltages required by other electronic components.
The use of custom designed converters allowed additional ripple filtering for low noise."""
            ))

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
