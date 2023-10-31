import pathlib
import matplotlib.pyplot as plt
import aastex
from . import preamble
# from . import variables
from . import authors
from . import sections

__all__ = [
    "document",
]

path_base = pathlib.Path(__file__).parent
path_pdf = path_base / 'instrument'
path_figures = path_base / 'figures'


def document() -> aastex.Document:

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 9
    plt.rcParams['lines.linewidth'] = 1

    doc = aastex.Document(
        default_filepath=path_pdf,
        documentclass='aastex631',
        document_options=[
            'twocolumn',
            # 'linenumbers',
        ]
    )

    doc.packages.append(aastex.Package('paralist'))
    doc.packages.append(aastex.Package('amsmath'))
    doc.packages.append(aastex.Package('acronym'))

    doc.preamble += preamble()

    # variables.append_to_document(doc)

    doc.append(aastex.Title('The EUV Snapshot Imaging Spectrograph'))

    doc += authors()

    # doc.append(sections.abstract.section())

    doc.append(sections.introduction.section())

    # doc.append(sections.esis_concept.section())
    #
    # doc.append(sections.science_objectives.section())
    #
    # doc.append(sections.esis_instrument.section(doc))
    #
    # doc.append(sections.mission_profile.section())
    #
    # doc.append(sections.conclusion.section())

    doc.append(aastex.Bibliography("sources"))

    return doc
