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
    doc.append(pylatex.Command('author', 'Roy Smart'))

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
