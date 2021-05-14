import pathlib
import matplotlib.pyplot as plt
import astropy.units as u
import pylatex
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

    doc.append(pylatex.Command('title', 'The EUV Snapshot Imaging Spectrograph'))

    str_msu = 'Montana State University, Department of Physics, P.O. Box 173840, Bozeman, MT 59717, USA'
    affil_msu = pylatex.Command('affiliation', str_msu)

    str_msfc = 'NASA Marshall Space Flight Center, Huntsville, AL 35812, USA'
    affil_msfc = pylatex.Command('affiliation', str_msfc)

    str_lbnl = 'Lawrence Berkeley National Laboratory, 1 Cyclotron Road, Berkeley, CA 94720, USA'
    affil_lbnl = pylatex.Command('affiliation', str_lbnl)

    str_rxo = 'Reflective X-ray Optics LLC, 425 Riverside Dr., #16G, New York, NY 10025, USA'
    affil_rxo = pylatex.Command('affiliation', str_rxo)


    doc.append(pylatex.Command('author', 'Hans T. Courrier'))
    doc.append(affil_msu)

    doc.append(pylatex.Command('author', 'Roy T. Smart'))
    doc.append(affil_msu)

    doc.append(pylatex.Command('author', 'Charles C. Kankelborg'))
    doc.append(affil_msu)

    doc.append(pylatex.Command('author', 'Amy R. Winebarger'))
    doc.append(affil_msfc)

    doc.append(pylatex.Command('author', 'Ken Kobayashi'))
    doc.append(affil_msfc)

    doc.append(pylatex.Command('author', 'Brent Beabout'))
    doc.append(affil_msfc)

    doc.append(pylatex.Command('author', 'Dyana Beabout'))
    doc.append(affil_msfc)

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

