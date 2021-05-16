import pathlib
import matplotlib.pyplot as plt
import astropy.units as u
import pylatex
import kgpy.latex
import esis.optics
import esis.science.papers.instrument.figures as figures

path_base = pathlib.Path(__file__).parent
path_pdf = path_base / 'instrument'
path_figures = path_base / 'figures'


def document() -> pylatex.Document:

    doc = pylatex.Document(
        default_filepath=str(path_pdf),
        documentclass='aastex63',
    )

    doc.packages.append(pylatex.Package('savesym'))
    doc.preamble.append(pylatex.NoEscape(
        r"""
        \savesymbol{tablenum}
        \usepackage{siunitx}
        \restoresymbol{SIX}{tablenum}
        """
    ))

    doc.preamble.append(pylatex.NoEscape(r'\newcommand{\amy}[1]{{{\color{red} #1}}}'))
    doc.preamble.append(pylatex.NoEscape(r'\newcommand{\jake}[1]{{{\color{purple} #1}}}'))

    doc.append(kgpy.latex.Title('The EUV Snapshot Imaging Spectrograph'))

    doc.append(pylatex.Command('author', 'Ben Carrol'))
    doc.append(affil_msu)

    doc.append(pylatex.Command('author', 'Jonathan Cirtain'))
    doc.append(affil_msfc)

    doc.append(pylatex.Command('author', 'James A. Duffy'))
    doc.append(affil_msfc)

    doc.append(pylatex.Command('author', 'Carlos Gomez'))
    doc.append(affil_msfc)

    doc.append(pylatex.Command('author', 'Eric Gullikson'))
    doc.append(affil_lbnl)

    doc.append(pylatex.Command('author', 'Micah Johnson'))
    doc.append(affil_msu)

    doc.append(pylatex.Command('author', 'Jacob D. Parker'))
    doc.append(affil_msu)

    doc.append(pylatex.Command('author', 'Laurel Rachmeler'))
    doc.append(affil_msfc)

    doc.append(pylatex.Command('author', 'Larry Springer'))
    doc.append(affil_msu)

    doc.append(pylatex.Command('author', 'David L. Windt'))
    doc.append(affil_rxo)

    with doc.create(pylatex.Section('Introduction')):
        pass

    with doc.create(pylatex.Section('The ESIS Concept')):

        with doc.create(pylatex.Subsection('Limitations of the MOSES Design')):

            pass

        with doc.create(pylatex.Subsection('ESIS Features')):

            with doc.create(pylatex.Figure(position='ht')) as esis_figure_3d:
                esis_figure_3d.add_image('figures/layout', width=pylatex.NoEscape(r'\textwidth'))

            with doc.create(pylatex.Figure(position='ht')) as esis_figure_3d:
                esis_figure_3d.add_image(str(figures.layout_pdf()), width=None)

    doc.append('text')

    return doc


if __name__ == '__main__':
    # plt.rcParams['axes.labelsize'] = 9
    # plt.rcParams['xtick.labelsize'] = 9
    # plt.rcParams['ytick.labelsize'] = 9
    # plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['font.size'] = 9
    plt.rcParams['lines.linewidth'] = 1

    doc = document()
    doc.generate_tex()
    doc.generate_pdf()

