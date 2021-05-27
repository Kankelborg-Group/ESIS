import pathlib
import matplotlib.pyplot as plt
import astropy.units as u
import pylatex
import kgpy.latex
import kgpy.units
import esis.optics
import esis.science.papers.instrument.figures as figures

path_base = pathlib.Path(__file__).parent
path_pdf = path_base / 'instrument'
path_figures = path_base / 'figures'


def document() -> kgpy.latex.Document:

    doc = kgpy.latex.Document(
        default_filepath=str(path_pdf),
        documentclass='aastex63',
    )

    doc.packages.append(pylatex.Package('savesym'))
    doc.preamble.append(pylatex.NoEscape(
        '\\savesymbol{tablenum}'
        '\\usepackage{siunitx}'
        '\\restoresymbol{SIX}{tablenum}'
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

    with doc.create(kgpy.latex.Abstract()):
        doc.append(pylatex.NoEscape(
            r"""The Extreme ultraviolet Snapshot Imaging Spectrograph (ESIS) is a next generation rocket borne 
            instrument that will investigate magnetic reconnection and energy transport in the solar atmosphere 
            \amy{by observing emission lines formed in the chromosphere (He\textsc{i} \SI{58.4}{\nano\meter}), the 
            transition region (O\,\textsc{v} \SI{62.9}{\nano\meter}), and corona 
            (Mg\,\textsc{x} \SI{62.5}{\nano\meter}).}
            \jake{JDP: Would make more sense to talk about the brighter Mg line?  609.8}
            The instrument is a pseudo Gregorian telescope; from prime focus, an array of spherical diffraction gratings 
            re-image with differing dispersion angles. 
            \amy{The instrument is a pseudo Gregorian telescope with an octagonal field stop at prime focus.  
            This field stop is re-imaged  using an array of four spherical diffraction gratings with differing 
            dispersion angles relative to ...? [ I want to say relative to solar north or field stop north or 
            something], with each diffraction grating projecting the spectrum onto a unique detector.}
            The slitless multi-projection design will obtain co-temporal spatial (\avgPlateScale) and 
            spectral (\dispersion) images at high cadence ($>=$\minCadence). 
            \amy{The instrument is designed to be capable of obtaining co-temporal spatial 
            (\avgPlateScale) and spectral (\dispersion) images at high cadence 
            ($>=$\minCadence).}
            \amy{Combining the co-temporal exposures from all the detectors will enable us to reconstruct line profile 
            information at high spatial and spectral resolution over a large (\fov) field of view. 
            The instrument was launched on September 30, 2019.  The flight data is described in a subsequent paper. }
            A single exposure will enable us to reconstruct line profile information at high spatial and spectral 
            resolution over a large (\fov) field of view. 
            The instrument is currently in the build up phase prior to spacecraft integration, testing, and launch."""
        ))

    with doc.create(pylatex.Section('Introduction', label="section:intro")):
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
    doc.generate_pdf()
    doc.generate_tex()
