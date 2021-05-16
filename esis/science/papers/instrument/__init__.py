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

            # with doc.create(pylatex.Figure(position='ht')) as esis_figure_3d:
            #     esis_figure_3d.add_image(str(figures.layout_pdf()), width=None)

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
    doc.generate_pdf()
    doc.generate_tex()

