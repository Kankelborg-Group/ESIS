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

    doc.append(sections.abstract.section())

    doc.append(sections.introduction.section())

    doc.append(sections.esis_concept.section())

    doc.append(sections.science_objectives.section())

    doc.append(sections.esis_instrument.section(doc))

    doc.append(sections.mission_profile.section())

    doc.append(sections.conclusion.section())

    doc.append(pylatex.Command('bibliography', arguments='sources'))

    return doc
