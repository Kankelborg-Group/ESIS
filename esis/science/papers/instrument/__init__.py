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

    doc.append(sections.esis_instrument.section(doc))

    doc.append(sections.mission_profile.section())

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
