import pathlib
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
import pylatex
import num2words
import kgpy.format
import kgpy.latex
import kgpy.units
import kgpy.chianti
import kgpy.optics
import esis.optics
import esis.science.papers.instrument.figures as figures
from . import optics

path_base = pathlib.Path(__file__).parent
path_pdf = path_base / 'instrument'
path_figures = path_base / 'figures'


def document() -> kgpy.latex.Document:

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 9
    plt.rcParams['lines.linewidth'] = 1

    wavl_digits = 2

    doc = kgpy.latex.Document(
        default_filepath=str(path_pdf),
        documentclass='aastex631',
        document_options=[
            'twocolumn',
            # 'linenumbers',
        ]
    )

    doc.packages.append(pylatex.Package('paralist'))
    doc.packages.append(pylatex.Package('acronym'))
    doc.packages.append(pylatex.Package('savesym'))
    doc.preamble.append(pylatex.NoEscape(
        '\\savesymbol{tablenum}\n'
        '\\usepackage{siunitx}\n'
        '\\restoresymbol{SIX}{tablenum}\n'
    ))

    doc.preamble.append(pylatex.NoEscape(
        r"""
\makeatletter
\newcommand{\acposs}[1]{%
 \expandafter\ifx\csname AC@#1\endcsname\AC@used
   \acs{#1}'s%
 \else
   \aclu{#1}'s (\acs{#1}'s)%
 \fi
}
\newcommand{\Acposs}[1]{%
 \expandafter\ifx\csname AC@#1\endcsname\AC@used
   \acs{#1}'s%
 \else
   \Aclu{#1}'s (\acs{#1}'s)%
 \fi
}
\makeatother"""
    ))

    doc.set_variable(
        name='ie',
        value=pylatex.NoEscape(r'\textit{i.e.}')
    )

    doc.set_variable(
        name='eg',
        value=pylatex.NoEscape(r'\textit{e.g.}')
    )

    doc.preamble.append(pylatex.NoEscape(r'\newcommand{\amy}[1]{{{\color{red} #1}}}'))
    doc.preamble.append(pylatex.NoEscape(r'\newcommand{\jake}[1]{{{\color{purple} #1}}}'))
    doc.preamble.append(pylatex.NoEscape(r'\newcommand{\roy}[1]{{{\color{blue} #1}}}'))

    doc.preamble.append(pylatex.Command('bibliographystyle', 'aasjournal'))
    doc.set_variable('spiejatis', pylatex.NoEscape(r'J~.Atmos.~Tel. \& Img.~Sys.'))

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

    requirements = esis.optics.design.requirements()

    optics_single = optics.as_designed_single_channel()

    optics_all = esis.flight.optics.as_measured()

    doc.set_variable_quantity(
        name='spatialResolutionRequirement',
        value=requirements.resolution_spatial,
        digits_after_decimal=1,
    )

    doc.set_variable_quantity(
        name='angularResolutionRequirement',
        value=requirements.resolution_angular,
        digits_after_decimal=0,
    )

    doc.set_variable_quantity(
        name='spectralResolutionRequirement',
        value=requirements.resolution_spectral,
        digits_after_decimal=0,
    )

    doc.set_variable_quantity(
        name='fovRequirement',
        value=requirements.fov,
        digits_after_decimal=0,
    )

    doc.set_variable_quantity(
        name='snrRequirement',
        value=requirements.snr,
        digits_after_decimal=1,
    )

    doc.set_variable_quantity(
        name='cadenceRequirement',
        value=requirements.cadence,
        digits_after_decimal=0,
    )

    doc.set_variable_quantity(
        name='observingTimeRequirement',
        value=requirements.length_observation,
        digits_after_decimal=0,
    )

    doc.set_variable_quantity(
        name='minWavelength',
        value=optics_single.wavelength_min,
        digits_after_decimal=wavl_digits,
    )

    doc.set_variable_quantity(
        name='maxWavelength',
        value=optics_single.wavelength_max,
        digits_after_decimal=wavl_digits,
    )

    doc.set_variable(
        name='numEmissionLines',
        value=str(optics_single.num_emission_lines),
    )

    doc.set_variable(
        name='chiantiDEM',
        value=pylatex.NoEscape(pylatex.Command('texttt', kgpy.chianti.dem_qs_file).dumps())
    )

    doc.set_variable(
        name='chiantiAbundances',
        value=pylatex.NoEscape(pylatex.Command('texttt', kgpy.chianti.abundance_qs_tr_file).dumps())
    )

    doc.set_variable_quantity(
        name='chiantiPressure',
        value=kgpy.chianti.pressure_qs,
        digits_after_decimal=0,
        scientific_notation=True,
    )

    wavelength = optics_single.bunch.wavelength
    ion = kgpy.chianti.to_spectroscopic(optics_single.bunch.ion)

    index_o5 = np.nonzero(optics_single.bunch.ion == 'o_5')[0][0]
    wavelength_o5 = wavelength[index_o5]
    doc.set_variable_quantity(
        name='OVwavelength',
        value=wavelength_o5,
        digits_after_decimal=wavl_digits,
    )
    doc.set_variable(
        name='OVion',
        value=pylatex.NoEscape(ion[index_o5])
    )
    doc.set_variable(
        name='OV',
        value=pylatex.NoEscape(r'\OVion\ \OVwavelength')
    )

    index_he1 = np.nonzero(optics_single.bunch.ion == 'he_1')[0][0]
    doc.set_variable_quantity(
        name='HeIwavelength',
        value=wavelength[index_he1],
        digits_after_decimal=wavl_digits,
    )
    doc.set_variable(
        name='HeIion',
        value=pylatex.NoEscape(ion[index_he1])
    )
    doc.set_variable(
        name='HeI',
        value=pylatex.NoEscape(r'\HeIion\ \HeIwavelength')
    )

    index_mg10 = np.nonzero(optics_single.bunch.ion == 'mg_10')[0][0]
    doc.set_variable_quantity(
        name='MgXwavelength',
        value=wavelength[index_mg10],
        digits_after_decimal=wavl_digits,
    )
    doc.set_variable(
        name='MgXion',
        value=pylatex.NoEscape(ion[index_mg10])
    )
    doc.set_variable(
        name='MgX',
        value=pylatex.NoEscape(r'\MgXion\ \MgXwavelength')
    )

    index_mg10_2 = np.nonzero(optics_single.bunch.ion == 'mg_10')[0][1]
    wavelength_mg10_2 = wavelength[index_mg10_2]
    doc.set_variable_quantity(
        name='MgXdimWavelength',
        value=wavelength_mg10_2,
        digits_after_decimal=wavl_digits,
    )
    doc.set_variable(
        name='MgXdimIon',
        value=pylatex.NoEscape(ion[index_mg10_2])
    )
    doc.set_variable(
        name='MgXdim',
        value=pylatex.NoEscape(r'\MgXdimIon\ \MgXdimWavelength')
    )

    doc.set_variable(
        name='numChannels',
        value=str(optics_all.num_channels),
    )

    doc.set_variable(
        name='numChannelsWords',
        value=num2words.num2words(optics_all.num_channels)
    )

    doc.set_variable(
        name='NumChannelsWords',
        value=num2words.num2words(optics_all.num_channels).capitalize()
    )

    doc.set_variable_quantity(
        name='magnification',
        value=optics_single.magnification.quantity,
        digits_after_decimal=3,
    )

    doc.set_variable_quantity(
        name='anamorphicMagnification',
        value=optics_single.grating.magnification_anamorphic,
    )

    dr = optics_single.detector.cylindrical_radius - optics_single.grating.cylindrical_radius
    dz = optics_single.detector.piston - optics_single.grating.piston
    doc.set_variable_quantity(
        name='tiltMagnification',
        value=1 / np.cos(optics_single.detector.inclination + np.arctan(dr / dz))
    )

    doc.set_variable_quantity(
        name='backFocalLength',
        value=optics_single.back_focal_length,
        digits_after_decimal=0,
    )

    doc.set_variable_quantity(
        name='radiusRatio',
        value=optics_single.radius_ratio,
    )

    doc.set_variable_quantity(
        name='armRatio',
        value=optics_single.arm_ratio,
    )

    doc.set_variable_quantity(
        name='fov',
        value=optics_single.field_of_view.quantity.mean().to(u.arcmin),
        digits_after_decimal=1,
    )

    doc.set_variable_quantity(
        name='plateScale',
        value=optics_single.plate_scale.quantity,
        digits_after_decimal=3,
    )

    doc.set_variable_quantity(
        name='spatialResolution',
        value=optics_single.resolution_spatial.quantity,
        digits_after_decimal=2,
    )

    doc.set_variable_quantity(
        name='dispersion',
        value=optics_single.dispersion.to(kgpy.units.mAA / u.pix),
        digits_after_decimal=1,
    )

    doc.set_variable_quantity(
        name='dispersionDoppler',
        value=optics_single.dispersion_doppler.to(u.km / u.s / u.pix),
        digits_after_decimal=1,
    )

    doc.set_variable_quantity(
        name='minCadence',
        value=optics_single.detector.exposure_length_min,
        digits_after_decimal=1,
    )

    doc.set_variable_quantity(
        name='primaryDiameter',
        value=2 * optics_single.primary.clear_half_width,
        digits_after_decimal=0,
    )

    doc.set_variable_quantity(
        name='primaryFocalLength',
        value=optics_single.primary.focal_length,
        digits_after_decimal=1,
    )

    doc.set_variable(
        name='primaryCoatingMaterial',
        value=pylatex.NoEscape('\\' + optics_single.primary.material.main.material[0]),
    )

    doc.set_variable(
        name='primaryCoatingMaterialShort',
        value=pylatex.NoEscape('\\' + optics_single.primary.material.main.material[0] + 'Short'),
    )

    doc.set_variable_quantity(
        name='primaryCoatingThickness',
        value=optics_single.primary.material.main.thickness[0],
        digits_after_decimal=0,
    )

    doc.set_variable(
        name='primaryCoatingBaseMaterial',
        value=pylatex.NoEscape('\\' + optics_single.primary.material.base.material[0]),
    )

    doc.set_variable_quantity(
        name='primaryCoatingBaseThickness',
        value=optics_single.primary.material.base.thickness[0],
        digits_after_decimal=0,
    )

    rays_o5 = kgpy.optics.rays.Rays(wavelength=wavelength_o5)
    doc.set_variable_quantity(
        name='primaryEfficiency',
        value=optics_all.primary.material.transmissivity(rays_o5).to(u.percent),
        digits_after_decimal=0,
    )

    doc.set_variable_quantity(
        name='primaryWitnessMeasurementIncidenceAngle',
        value=esis.optics.primary.efficiency.witness.angle_input,
        digits_after_decimal=0,
    )

    doc.set_variable(
        name='primaryMeasurementDate',
        value=esis.optics.primary.efficiency.witness.date_measurement.strftime('%Y %B %d'),
    )

    doc.set_variable_quantity(
        name='fieldStopDiameter',
        value=optics_single.field_stop.clear_width,
        digits_after_decimal=3,
    )

    doc.set_variable_quantity(
        name='gratingHeight',
        value=optics_single.grating.height,
        digits_after_decimal=1,
    )

    doc.set_variable_quantity(
        name='gratingShortWidth',
        value=optics_single.grating.width_short,
        digits_after_decimal=1,
    )

    doc.set_variable_quantity(
        name='gratingLongWidth',
        value=optics_single.grating.width_long,
        digits_after_decimal=1,
    )

    doc.set_variable_quantity(
        name='gratingRulingSpacing',
        value=optics_single.grating.surface.rulings.ruling_spacing.to(u.um),
    )

    doc.set_variable_quantity(
        name='gratingLinearRulingSpacingCoefficient',
        value=optics_single.grating.ruling_spacing_coeff_linear,
        scientific_notation=True,
    )

    doc.set_variable_quantity(
        name='gratingQuadraticRulingSpacingCoefficient',
        value=optics_single.grating.ruling_spacing_coeff_quadratic,
        scientific_notation=True,
    )

    doc.set_variable_quantity(
        name='gratingRadius',
        value=optics_single.grating.tangential_radius,
    )

    doc.set_variable_quantity(
        name='gratingInputAngle',
        value=optics_single.grating.nominal_input_angle,
    )

    doc.set_variable_quantity(
        name='gratingOutputAngle',
        value=optics_single.grating.nominal_output_angle,
    )

    doc.set_variable(
        name='firstGratingCoatingMaterial',
        value=pylatex.NoEscape(f'\\{optics_single.grating.material.main.material[0]}')
    )
    doc.set_variable(
        name='firstGratingCoatingMaterialShort',
        value=pylatex.NoEscape(f'\\{optics_single.grating.material.main.material[0]}Short')
    )
    doc.set_variable(
        name='secondGratingCoatingMaterial',
        value=pylatex.NoEscape(f'\\{optics_single.grating.material.main.material[1]}')
    )
    doc.set_variable(
        name='secondGratingCoatingMaterialShort',
        value=pylatex.NoEscape(f'\\{optics_single.grating.material.main.material[1]}Short')
    )
    doc.set_variable(
        name='thirdGratingCoatingMaterial',
        value=pylatex.NoEscape(f'\\{optics_single.grating.material.main.material[2]}')
    )
    doc.set_variable(
        name='thirdGratingCoatingMaterialShort',
        value=pylatex.NoEscape(f'\\{optics_single.grating.material.main.material[2]}Short')
    )
    doc.set_variable(
        name='gratingCoatingMaterial',
        value=pylatex.NoEscape(
            r'\firstGratingCoatingMaterial, \secondGratingCoatingMaterial, and \thirdGratingCoatingMaterial')
    )
    doc.set_variable(
        name='gratingCoatingMaterialShort',
        value=pylatex.NoEscape(
            r'\firstGratingCoatingMaterialShort/\secondGratingCoatingMaterialShort/\thirdGratingCoatingMaterialShort'),
    )

    grating_nlayers = optics_single.grating.material.cap.num_periods + optics_single.grating.material.main.num_periods
    doc.set_variable(
        name='gratingCoatingNumLayers',
        value=str(grating_nlayers)
    )
    doc.set_variable(
        name='gratingCoatingNumLayersWords',
        value=num2words.num2words(grating_nlayers),
    )

    grating_efficiency = optics_all.grating.material.transmissivity(rays_o5).to(u.percent)
    doc.set_variable_quantity(
        name='gratingEfficiency',
        value=grating_efficiency,
        digits_after_decimal=0,
    )

    grating_witness_efficiency = optics_all.grating.witness.transmissivity(rays_o5).to(u.percent)
    doc.set_variable_quantity(
        name='gratingWitnessEfficiency',
        value=grating_witness_efficiency,
        digits_after_decimal=0,
    )

    doc.set_variable_quantity(
        name='gratingGrooveEfficiency',
        value=(grating_efficiency / grating_witness_efficiency).to(u.percent),
        digits_after_decimal=0,
    )

    is_number = esis.optics.grating.efficiency.manufacturing_number == optics_all.grating.manufacturing_number
    doc.set_variable(
        name='testGratingChannelIndex',
        value=optics_all.channel_name[np.nonzero(is_number)][0]
    )

    doc.set_variable(
        name='testGratingDate',
        value=esis.optics.grating.efficiency.date_measurement.strftime('%Y %B %d')
    )

    doc.set_variable_quantity(
        name='gratingWitnessMeasurementIncidenceAngle',
        value=esis.optics.grating.efficiency.witness.angle_input,
        digits_after_decimal=0,
    )

    doc.set_variable_quantity(
        name='gratingMeasurementIncidenceAngle',
        value=esis.optics.grating.efficiency.vs_wavelength()[0],
        digits_after_decimal=0,
    )

    doc.set_variable_quantity(
        name='gratingTestWavelength',
        value=esis.optics.grating.efficiency.wavelength_nominal,
        digits_after_decimal=0,
    )

    doc.set_variable_quantity(
        name='filterDiameter',
        value=optics_single.filter.clear_diameter,
        digits_after_decimal=1,
    )

    doc.set_variable_quantity(
        name='filterThickness',
        value=optics_single.filter.thickness,
        digits_after_decimal=0,
    )

    doc.set_variable_quantity(
        name='filterOxideThickness',
        value=optics_single.filter.thickness_oxide,
        digits_after_decimal=0,
    )

    doc.set_variable(
        name='filterMaterial',
        value=optics_single.filter.surface.material.name,
    )

    doc.set_variable_quantity(
        name='filterMeshRatio',
        value=optics_single.filter.mesh_ratio,
        digits_after_decimal=0,
    )

    doc.set_variable(
        name='filterMeshMaterial',
        value=pylatex.NoEscape(f'\\{optics_single.filter.mesh_material}'),
    )

    doc.set_variable(
        name='filterMeshMaterialShort',
        value=pylatex.NoEscape(f'\\{optics_single.filter.mesh_material}Short'),
    )

    doc.set_variable_quantity(
        name='filterEfficiency',
        value=optics_single.filter.surface.material.transmissivity(rays_o5).to(u.percent),
        digits_after_decimal=0,
    )

    doc.set_variable(
        name='detectorName',
        value=str(optics_single.detector.name),
    )

    doc.set_variable(
        name='detectorPixelsX',
        value=str(optics_single.detector.num_pixels[0]),
    )

    doc.set_variable(
        name='detectorPixelsY',
        value=str(optics_single.detector.num_pixels[1]),
    )

    doc.set_variable_quantity(
        name='detectorPixelSize',
        value=optics_single.detector.pixel_width,
        digits_after_decimal=0,
    )

    doc.set_variable_quantity(
        name='detectorQuantumEfficiency',
        value=optics_single.detector.surface.material.transmissivity(rays_o5).to(u.percent),
        digits_after_decimal=0
    )

    doc.set_variable(
        name='defaultPupilSamples',
        value=figures.kwargs_optics_default['pupil_samples'],
    )

    doc.set_variable(
        name='defaultFieldSamples',
        value=figures.kwargs_optics_default['field_samples']
    )

    doc.set_variable(
        name='defaultNumEmissionLines',
        value=figures.num_emission_lines_default,
    )

    doc.set_variable(
        name='defaultNumEmissionLinesWords',
        value=num2words.num2words(figures.num_emission_lines_default),
    )

    doc.set_variable(
        name='psfPupilSamples',
        value=figures.psf_pupil_samples,
    )

    doc.set_variable(
        name='psfFieldSamples',
        value=figures.psf_field_samples,
    )

    doc.preamble.append(kgpy.latex.Acronym('ESIS', r'EUV Snapshot Imaging Spectrograph'))
    doc.preamble.append(kgpy.latex.Acronym('MOSES', r'Multi-order Solar EUV Spectrograph'))
    doc.preamble.append(kgpy.latex.Acronym('TRACE', r'Transition Region and Coronal Explorer'))
    doc.preamble.append(kgpy.latex.Acronym('AIA', r'Atmospheric Imaging Assembly'))
    doc.preamble.append(kgpy.latex.Acronym('IRIS', r'Interface Region Imaging Spectrograph'))
    doc.preamble.append(kgpy.latex.Acronym('EIS', 'EUV Imaging Spectrograph'))
    doc.preamble.append(kgpy.latex.Acronym('EUNIS', 'EUV Normal-incidence Spectrometer'))
    doc.preamble.append(kgpy.latex.Acronym('CDS', 'Coronal Diagnostic Spectrometer', short=True))
    doc.preamble.append(kgpy.latex.Acronym('MDI', r'Michelson Doppler Imager'))
    doc.preamble.append(kgpy.latex.Acronym('FUV', 'far ultraviolet'))
    doc.preamble.append(kgpy.latex.Acronym('EUV', 'extreme ultraviolet'))
    doc.preamble.append(kgpy.latex.Acronym('TR', 'transition region'))
    doc.preamble.append(kgpy.latex.Acronym('CTIS', 'computed tomography imaging spectrograph', plural=True))
    doc.preamble.append(kgpy.latex.Acronym('FOV', 'field of view'))
    doc.preamble.append(kgpy.latex.Acronym('SNR', 'signal-to-noise ratio'))
    doc.preamble.append(kgpy.latex.Acronym('PSF', 'point-spread function', plural=True))
    doc.preamble.append(kgpy.latex.Acronym('NRL', 'Naval Research Laboratory'))
    doc.preamble.append(kgpy.latex.Acronym('MHD', 'magnetohydrodynamic'))
    doc.preamble.append(kgpy.latex.Acronym('EE', 'explosive event', plural=True))
    doc.preamble.append(kgpy.latex.Acronym('QS', 'quiet sun'))
    doc.preamble.append(kgpy.latex.Acronym('AR', 'active region'))
    doc.preamble.append(kgpy.latex.Acronym('CH', 'coronal hole'))
    doc.preamble.append(kgpy.latex.Acronym('CCD', 'charge-coupled device', plural=True))
    doc.preamble.append(kgpy.latex.Acronym('ULE', 'ultra-low expansion'))
    doc.preamble.append(kgpy.latex.Acronym('EDM', 'electrical discharge machining'))
    doc.preamble.append(kgpy.latex.Acronym('DEM', 'differential emission measure'))
    doc.preamble.append(kgpy.latex.Acronym('MTF', 'modulation transfer function'))
    doc.preamble.append(kgpy.latex.Acronym('LBNL', 'Lawrence Berkley National Laboratory'))

    doc.preamble.append(kgpy.latex.Acronym('SiC', 'silicon carbide', short=True))
    doc.preamble.append(kgpy.latex.Acronym('Al', 'aluminum', short=True))
    doc.preamble.append(kgpy.latex.Acronym('Mg', 'magnesium', short=True))
    doc.preamble.append(kgpy.latex.Acronym('Si', 'silicon', short=True))
    doc.preamble.append(kgpy.latex.Acronym('Cr', 'chromium', short=True))
    doc.preamble.append(kgpy.latex.Acronym('Ni', 'nickel', short=True))

    doc.preamble.append(pylatex.Command('DeclareSIUnit', [pylatex.NoEscape(r'\angstrom'), pylatex.NoEscape(r'\AA')]))

    with doc.create(kgpy.latex.Abstract()):
        doc.append(pylatex.NoEscape(
            r"""The \ESIS\ is a next generation rocket borne instrument that will investigate magnetic reconnection 
and energy transport in the solar atmosphere 
\amy{by observing emission lines formed in the chromosphere (\HeI), the transition region (\OV), and corona (\MgX).}
\jake{JDP: Would make more sense to talk about the brighter Mg line?  609.8}
The instrument is a pseudo Gregorian telescope; 
from prime focus, an array of spherical diffraction gratings re-image with differing dispersion angles. 
\amy{The instrument is a pseudo Gregorian telescope with an octagonal field stop at prime focus.  
This field stop is re-imaged  using an array of \numChannelsWords\ spherical diffraction gratings with differing 
ispersion angles relative to ...? [ I want to say relative to solar north or field stop north or something], with each 
diffraction grating projecting the spectrum onto a unique detector.}
The slitless multi-projection design will obtain co-temporal spatial (\plateScale) and spectral (\dispersion) images 
at high cadence ($>=$\minCadence). 
\amy{The instrument is designed to be capable of obtaining co-temporal spatial (\plateScale) and spectral 
(\dispersion) images at high cadence ($>=$\minCadence).}
\amy{Combining the co-temporal exposures from all the detectors will enable us to reconstruct line profile information 
at high spatial and spectral resolution over a large (\fov) \FOV. 
The instrument was launched on September 30, 2019.  The flight data is described in a subsequent paper. }
A single exposure will enable us to reconstruct line profile information at high spatial and spectral resolution over a 
large (\fov) \FOV. 
The instrument is currently in the build up phase prior to spacecraft integration, testing, and launch.
\acresetall"""
        ))

    with doc.create(pylatex.Section('Introduction')):
        doc.append(pylatex.NoEscape(
            r"""The solar atmosphere, as viewed from space in its characteristic short wavelengths (\FUV, \EUV, and soft 
X-ray), is a three-dimensional scene evolving in time: $I[x,y,\lambda,t]$.
Here the solar sky plane spatial coordinates, $x$ and $y$, and the wavelength axis, $\lambda$, comprise the three 
dimensions of the scene, while $t$ represents the temporal axis.
An ideal instrument would capture a spatial/spectral data cube ($I[x,y,\lambda]$) at a rapid temporal cadence ($t$), 
however, practical limitations lead us to accept various compromises of these four observables.
Approaching this ideal is the fast tunable filtergraph (\ie\ fast tunable Fabry--P\'erot etalons, \eg\ the GREGOR 
Fabry--P{\'e}rot Interferometer, \citep{Puschmann12}), but the materials do not exist to extend this technology to 
\EUV\ wavelengths shortward of $\sim$\SI{150}{\nano\meter}~\citep{2000WuelserFP}.
Imagers like the \TRACE~\citep{Handy99} and the \AIA~\citep{Lemen12} collect high cadence 2D \EUV\ spatial scenes, but 
they collect spectrally integrated intensity over a fixed passband that is not narrow enough to isolate a single 
emission line.  
In principle, filter ratios that make use of spectrally adjacent multilayer \EUV\ passbands could detect Doppler 
shifts~\citep{Sakao99}.
However, the passbands of the multilayer coatings are still wide enough that the presence of weaker contaminant lines 
limits resolution of Doppler shifts to $\sim$\SI{1000}{\kilo\meter\per\second}~\citep{Kobayashi00}.
Slit spectrographs (\eg\ the \IRIS~\citep{IRIS14}) obtain fast, high-resolution spatial and spectral observations, but are 
limited by the narrow \FOV\ of the spectrograph slit.
The $I[x,y,\lambda]$ data cube can be built up by rastering the slit pointing, but it cannot be co-temporal along the 
raster axis.
Moreover, extended and dynamic scenes can change significantly in the time required to raster over their extant.  

A different approach is to forego the entrance slit employed by traditional spectrographs entirely.
The \Acposs{NRL} SO82A~\citep{Tousey73,Tousey77} was one of the first instruments to pioneer this method.
The `overlappograms' obtained by SO82A identified several spectral line transitions~\citep{Feldman85}, and have more 
recently been used to determine line ratios in solar flares~\citep{Keenan06}.
Unfortunately, for closely spaced \EUV\ lines, the dispersed images from the single diffraction order suffer from 
considerable overlap confusion.
Image overlap is all but unavoidable with this configuration, however, overlappograms can be disentangled, or inverted, 
under the right circumstances.  

In analogy to a tomographic imaging problem~\citep{Kak88}, inversion of an overlapping spatial/spectral scene can be 
facilitated by increasing the number of independent spectral projections, or `look angles,' through the 3D 
$(x,y,\lambda)$ scene~\citep{DeForest04}.
For example, \citet{DeForest04} demonstrated recovery of Doppler shifts in magnetograms from two dispersed orders of a 
grating at the output of the \MDI~\citep{Scherrer95}.
The quality of the inversion (\eg\ recovery of higher order spectral line moments) can also be improved by additional 
projections~\citep{Kak88,Descour97}, generally at the cost of computational complexity~\citep{Hagen08}.
\Acp{CTIS}~\citep{okamoto1991,Bulygin91,Descour95} leverage this concept by obtaining multiple, simultaneous dispersed 
images of an object or scene; 
upwards of 25 grating diffraction orders may be projected onto a single detector plane~\citep{Descour97}.
Through post processing of these images, \CTIS\ can recover a 3D data cube from a (spectrally) smooth and continuous 
scene over a large bandpass (\eg\ \citet{Hagen08}).

The \MOSES~\citep{Fox10,Fox11} is our first effort aimed at developing the unique capability of simultaneous 
imaging and spectroscopy for solar \EUV\ scenes.
\MOSES\ is a three-order slitless spectrograph that seeks to combine the simplicity of the SO82A concept with the 
advantages of a \CTIS\ instrument.
A single diffraction grating (in conjunction with a fold mirror) projects the $m=\pm1$ and the un-dispersed $m=0$ order 
onto three different detectors.
Through a combination of dispersion and multi-layer coatings, the passband of the $m=\pm1$ orders encompasses only a few
solar \EUV\ emission lines.
Thus, \MOSES\ overlappograms consist of only a handful of spectral images.
This constraint on the volume of the 3D data cube helps make inversion of \MOSES\ data better-posed despite the 
discontinuous nature of the solar \EUV\ spectrum.
This working concept enabled by \MOSES\ has been proven over the course of two previous rocket flights.
Through inversion of \MOSES\ overlappograms, \citet{Fox10} obtained unprecedented measurements of Doppler shifts 
(\ie\ line widths) of \TR\ explosive events as a function of time and space while \citet{Rust17} recovered splitting and 
distinct moments of compact \TR\ bright point line profiles.

Building on the working concept demonstrated by \MOSES, here we describe a new instrument, the \ESIS, that will improve 
on past efforts to produce a solar \EUV\ spectral map.
\ESIS\ will fly alongside \MOSES\ and will observe the \TR\ and corona of the solar atmosphere in the \OV\ and \MgX\ / 
\MgXdimWavelength\ spectral lines.
In Section~\ref{sec:TheESISConcept} we detail how our experience with the \MOSES\ instrument has shaped the design of 
\ESIS.
Section~\ref{sec:ScienceObjectives} describes the narrow scientific objectives and the requirements placed on the new 
instrument.
Section~\ref{sec:TheESISInstrument} describes the technical approach to meet our scientific objectives, followed by a 
brief description of the expected mission profile in Section~\ref{sec:MissionProfile}.
The current status and progress toward launch is summarized in Section~\ref{sec:ConclusionsandOutlook}."""
        ))

    with doc.create(pylatex.Section('The ESIS Concept')):
        doc.append(pylatex.NoEscape(
            r"""A primary goal of the \ESIS\ instrument is to improve the implementation of \EUV\ snapshot imaging 
spectroscopy demonstrated by \MOSES.  
Therefore, the design of the new instrument draws heavily from experiences and lessons learned through two flights of 
the \MOSES\ instrument.
\ESIS\ and \MOSES\ are both slitless, multi-projection spectrographs.
As such, both produce dispersed images of a narrow portion of the solar spectrum, with the goal of enabling the 
reconstruction of a spectral line profile at every point in the field of view.
The similarities end there, however, as the optical layout of \ESIS\ differs significantly from that of \MOSES.
In this section, we detail some difficulties and limitations encountered with \MOSES, then describe how the new design 
of \ESIS\ addresses these issues."""
        ))
        with doc.create(pylatex.Subsection('Limitations of the MOSES Design')):
            doc.append(pylatex.NoEscape(
                r"""The \MOSES\ design features a single concave diffraction grating forming images on three \CCD\ 
detectors~\citep{Fox10} (Fig.~\ref{fig:mosesSchematic}). 
The optical path is folded in half by a single flat secondary mirror (omitted in Fig.~\ref{fig:mosesSchematic}).
Provided that the three cameras are positioned correctly, this arragement allows the entire telescope to be brought 
into focus using only the central (undispersed) order and a visible light source.
Unfortunately this design uses volume inefficiently for two reasons.
First, the lack of magnification by the secondary mirror limits the folded length of the entire telescope to be no less 
than half the \SI{5}{\meter} focal length of the grating~\citep{Fox10,Fox11}.
Second, the dispersion of the instrument is controlled by the placement of the cameras.
To achieve the maximum dispersion of \SI{29}{\kilo\meter\per\second}~\citep{Fox10}, the outboard orders are imaged as 
far apart as possible in the $\sim22''$ diameter cross section of the rocket payload.
The resulting planar dispersion poorly fills the cylindrical volume of the payload, leaving much unused space along the 
orthogonal planes."""
            ))

            with doc.create(pylatex.Figure(position='!ht')) as moses_schematic:
                moses_schematic.add_image('figures/old/MOSES_Schematic', width=pylatex.NoEscape(r'\columnwidth'))
                moses_schematic.add_caption(pylatex.NoEscape(
                    r"""Schematic diagram of the MOSES instrument.
Incident light on the right forms an undispersed image on the central $m=0$ \CCD.
Dispersed images are formed on the outboard $m=\pm1$ \CCDs."""
                ))
                moses_schematic.append(kgpy.latex.Label('fig:mosesSchematic'))

            doc.append(pylatex.NoEscape(
                r"""Furthermore, the monolithic secondary, though it confers the focus advantage noted above, does not 
allow efficient placement of the dispersed image order cameras.  
For all practical purposes, the diameter of the payload (\SI{0.56}{\meter}) can only accommodate three diffraction 
orders ($m=-1, 0, +1$).
Therefore, \textit{\MOSES\ can only collect, at most, three pieces of information at each point in the field of view.}
From this, it is not reasonable to expect the reconstruction of more than three degrees of freedom in this spectrum, 
except in the case very compact, isolated features such as those described by \citet{Fox10} and \citet{Rust17}.
Consequently, it is a reasonable approximation to say that \MOSES\ is sensitive primarily to spectral line intensities, 
shifts, and widths \citep{KankThom01}.
With any tomographic apparatus, the degree of detail that can be resolved in the object depends critically on the 
number of viewing angles~\citep{Kak88,Descour97,Hagen08}.
So it is with the spectrum we observe with \MOSES: more dispersed images are required to confer sensitivity to finer 
spectral details such additional lines in the passband or higher moments of the spectral line shape.

A related issue stems from the use of a single grating, with a single plane of dispersion.
Since the solar corona and transition region are structured by magnetic fields, the scene tends to be dominated by 
field aligned structures such as loops~\citep{Rosner78,Bonnet80}.
When the \MOSES\ dispersion direction happens to be aligned nearly perpendicular to the magnetic field, filamentary 
structures on the transition region serve almost as spectrograph slits unto themselves.
The estimation of Doppler shifts then becomes a simple act of triangulation, and broadenings are also readily 
diagnosed~\citep{Fox10,Courrier18}.
A double-peaked profile can also be observed with sufficiently isolated features~\citep{Rust17}.
Unfortunately, solar magnetic fields in the transition region are quite complex and do not have a global preferred 
direction.
In cases where the field is nearly parallel to the instrument dispersion, spectral shifts and broadenings are not 
readily apparent.

The single diffraction grating also leads to a compromise in the optical performance of the instrument. 
Since the \MOSES\ grating forms images in three orders simultaneously, aberration cannot be simultaneously optimized for 
all three of those spectral orders.
A result of this design is that the orientations (\ie\,the central axis) of the \PSFs\ vary order to order~\citep{Rust17}.
During the first mission, MOSES was flown with a small amount of defocus~\citep{Rust17}, which exacerbated the 
inter-order \PSF\ variation and caused the individual \PSFs\ to span several pixels~\citep{Rust17,Atwood18}.
The combination of these two effects results in spurious spectral features that require additional 
consideration~\citep{Atwood18} and further increase the complexity of the inversion process~\citep{Rust17,Courrier18}.
 
Another complication is that the spatial and spectral content differs slightly between the three \MOSES\ image orders.
This is because the \MOSES\ \FOV\ is defined by a combination of the aperture of the grating (\ie\,the entrance 
aperture of the telescope) and the spatial extent of the \CCDs.
The \FOV\ in the $m=\pm1$ orders is shifted along the dispersion axis as a function of wavelength, dependant upon where 
the dispersed spectral images intercept the $m=\pm1$ \CCDs.
Spatially, this effect is limited to only a handful of pixel columns at the edges of each image order.
Of higher concern is the `spectral contamination' allowed by this layout; 
\citet{Parker16} found that bright spectral lines and continuum far outside the wavelength passband and nominal $m=0$ 
\FOV\ could be diffracted onto the outboard order \CCDs.
This off-band contamination is detected as systematic intensity variation that lacks an anti-symmetric pairing in the 
opposite dispersed image order.
Analysis of the spectral contamination is ongoing. 

Finally, the exposure cadence of \MOSES\ is hindered by an $\sim$\SI{6}{\second} readout time for the 
\CCDs~\citep{Fox11}.
The observing interval for a solar sounding rocket flight is very short, typically about five minutes.
Consequently, every second of observing time is precious, both to achieve adequate exposure time and to catch the full 
development of dynamical phenomena.
The \MOSES\ observing duty cycle is $\sim$\SI{50}{\percent} since it is limited by the readout time of the \CCDs.
Thus, valuable observing time is lost.
The readout data gap impelled us to develop a \MOSES\ exposure sequence with exposures ranging from 
$0.25$-\SI{24}{\second}, a careful trade-off between deep and fast exposures. 

In summary, our experience leads us to conclude that the \MOSES\ design has the following primary limitations:
\begin{enumerate}
    \item inefficient use of volume \label{item-length} %(x and y direction)
    \item dispersion constrained by payload dimensions \label{item-disp_con}
    \item too few dispersed images (orders) \label{item-orders}
    \item single plane dispersion \label{item-dispersion}
    \item lack of aberration control \label{item-PSF}
    \item insufficiently defined FOV \label{item-FOV}
    \item sub-optimal exposure cadence \label{item-CAD}
\end{enumerate}
In designing \ESIS, we have sought to improve upon each of these points.
"""
            ))

        with doc.create(pylatex.Subsection('ESIS Features')):
            with doc.create(kgpy.latex.FigureStar(position='!ht')) as figure:
                figure.add_image('figures/old/layout', width=pylatex.NoEscape(r'\textwidth'))

            with doc.create(kgpy.latex.FigureStar(position='!ht')) as figure:
                figure.add_image(str(figures.layout_pdf()), width=None)
                figure.add_caption(pylatex.NoEscape(
                    r"""The \ESIS\ instrument is a pseudo-Gregorian design.
The secondary mirror is replaced by a segmented array of concave diffraction gratings.
The field stop at prime focus defines instrument spatial/spectral \FOV.
\CCDs\ are arrayed around the primary mirror, each associated with a particular grating.
Eight grating positions appear in this schematic; only six fit within the volume of the rocket payload.
\NumChannelsWords\ channels are populated for the first flight."""
                ))
                figure.append(kgpy.latex.Label('fig:layout'))

            doc.append(pylatex.NoEscape(
                r"""The layout of \ESIS\ (Fig.~\ref{fig:layout}) is a modified form of Gregorian telescope.
Incoming light is brought to focus at an octagonal field stop by a parabolic primary mirror.
In the \ESIS\ layout, the secondary mirror of a typical Gregorian telescope is replaced by a segmented, octagonal array 
of diffraction gratings.
From the field stop, the gratings re-image to \CCD\ detectors arranged radially around the primary mirror.
The gratings are blazed for first order, so that each \CCD\ is fed by a single corresponding grating, and all the 
gratings are identical in design.
The features of this new layout address all of the limitations described in 
Sect.~\ref{subsec:LimitationsoftheMOSESDesign}, and are summarized here.

Replacing the secondary mirror with an array of concave diffraction gratings confers several advantages to \ESIS\ over 
\MOSES.
First, the magnification of the \ESIS\ gratings results in a shorter axial length than MOSES, without sacrificing 
spatial or spectral resolution.
Second, the magnification and tilt of an individual grating controls the position of the dispersed image with respect 
to the optical axis, so that the spectral resolution is not as constrained by the payload dimensions.
Third, the radial symmetry of the design places the cameras closer together, resulting in a more compact instrument.
Furthermore, by arranging the detectors around the optical axis, more dispersed grating orders can be populated; 
up to eight gratings can be arrayed around the \ESIS\ primary mirror (up to six with the current optical table).
This contrasts the three image orders available in the planar symmetry of \MOSES.
Taken together, these three design features make \ESIS\ more compact than \MOSES\ 
(\S\,\ref{subsec:LimitationsoftheMOSESDesign} item~\ref{item-length}), improve spectral resolution 
(\S\,\ref{subsec:LimitationsoftheMOSESDesign} item~\ref{item-disp_con}) and allow the collection of more projections to 
better constrain the interpretation of the data (\S\,\ref{subsec:LimitationsoftheMOSESDesign} item~\ref{item-orders}).
 
The \ESIS\ gratings are arranged in a segmented array, clocked in \SI{45}{\degree} increments, so that there are 
\numChannelsWords\ distinct dispersion planes.
This will greatly aid in reconstructing spectral line profiles since the dispersion space of \ESIS\ occupies a 3D 
volume rather than a 2D plane as with \MOSES.
For \ESIS, there will always be a dispersion plane within \SI{22.5}{\degree} of the normal to any loop-like feature in 
the solar atmosphere.
As discussed in \S\,\ref{subsec:LimitationsoftheMOSESDesign}, a nearly perpendicular dispersion plane allows a 
filamentary structure to serve like a spectrographic slit, resulting in a clear presentation of the spectrum.
This feature addresses \S\,\ref{subsec:LimitationsoftheMOSESDesign} item~\ref{item-dispersion}. 

Rather than forming images at three spectral orders from a single grating, each \ESIS\ imaging channel has a dedicated 
grating.
Aberrations are controlled by optimizing the grating design to form images in first order, over a narrow range of ray 
deviation angles.
This design controls aberration well enough to allow pixel-limited imaging, avoiding the \PSF\ mismatch problems inherent 
to the \MOSES\ design (\S\,\ref{subsec:LimitationsoftheMOSESDesign} item \ref{item-PSF}).
A disadvantage of this arrangement is that \ESIS\ lacks a zero order focus.
In its flight configuration with gratings optimized around a \OVwavelength\ wavelength, the instrument cannot be 
aligned and focused in visible light like \MOSES.
Visible gratings and a special alignment transfer apparatus (\S\,\ref{subsec:AlignmentandFocus}) must be used for 
alignment and focus of \ESIS.

The \ESIS\ design also includes an octagonal field stop placed at prime focus.
This confers two advantages.
First, the field stop fully defines the instrument \FOV, so that \ESIS\ is not susceptible to the spectral contamination 
observed in \MOSES\ data (\S\,\ref{subsec:LimitationsoftheMOSESDesign} limitation~\ref{item-FOV}).
Second, each spectral image observed by \ESIS\ will be bordered by the outline of the field stop 
(\eg\,\S\,\ref{subsec:Optics}).
This aids the inversion process since outside of this sharp edge the intensity is zero for any look angle through an 
ESIS data cube.
Additionally, the symmetry of the field stop gives multiple checkpoints where the edge inversion is duplicated in the 
dispersed images produced by adjacent orders.
The size and octagonal shape of the field stop are defined by the requirement that all \CCDs\ must see the entire \FOV\ 
from edge to edge, while leaving a small margin for alignment. 

Lastly, in contrast to \MOSES, \ESIS\ employs frame transfer \CCDs\ to make optimum use of our five minutes of observing 
time.
The \ESIS\ design is shutterless, so that each detector is always integrating.
The result is a \SI{100}{\percent} duty cycle.
The lack of downtime for readout also allows \ESIS\ to operate at a fixed, rapid cadence of $\sim$\SI{3}{\second}.
Longer integration times can be achieved for faint features by exposure stacking 
(\S\,\ref{subsec:LimitationsoftheMOSESDesign} item~\ref{item-CAD}).

In summary, the \ESIS\ concept addresses all the limitations of the \MOSES\ design enumerated in 
\S\,\ref{subsec:LimitationsoftheMOSESDesign}.
The volume of the \ESIS\ optical layout is smaller than \MOSES\ by almost a factor of two, yet with a smaller \PSF, 
improved spectral resolution, and faster exposure cadence.
\ESIS\ offers several features to improve the recovery of spectral information, including more channels, crossed 
dispersion planes, and a field stop."""
            ))

    with doc.create(pylatex.Section('Science Objectives')):
        doc.append(pylatex.NoEscape(
            r"""The previous section discussed the qualitative design aspects of \ESIS\ learned from experience with the 
\MOSES\ instrument.  
\MOSES, in turn, demonstrated a working concept of simultaneous \EUV\ imaging and spectroscopy.
This concept adds a unique capability to the science that we can obtain from the \EUV\ solar atmosphere.
\ESIS, sharing the same payload volume as \MOSES, is manifested to fly in 2019.
In this section, we set forth specific scientific objectives for the combined \ESIS/\MOSES\ mission.
From these objectives, and with an eye toward synergistic operation of \MOSES\ and \ESIS, in 
\S\,\ref{subsec:ScienceRequirements} we derive the quantitative science requirements that drive the \ESIS\ design.

The combined \ESIS/\MOSES\ mission will address the following two overarching science goals: \begin{inparaenum}[(1)] 
\item observe magnetic reconnection in the \TR\label{item-goal1}, and \item map the transfer of energy through the \TR\ 
with emphasis on \MHD\ waves\label{item-goal2}. \end{inparaenum}
These objectives have significant overlap with the missions of \IRIS~\citep{IRIS14}, the \EIS~\citep{Culhane07}
aboard Hinode, the \EUNIS~\citep{Brosius07,Brosius14}, and a long history of FUV and EUV slit spectrographs.
The \ESIS\ instrument, however, can obtain both spatial and spectral information co-temporally.
This will allow us to resolve complicated morphologies of compact \TR\ reconnection events (as was done with 
\MOSES~\citep{Fox11,Rust17,Courrier18}) and observe signatures of \MHD\ waves over a large portion of the solar disk.
Therefore, in support of goal~\ref{item-goal1}, we will use \ESIS\ and \MOSES\ to map flows as a function of time and 
space in multiple \TR\ reconnection events.
To achieve goal~\ref{item-goal2}, we will cross-correlate the evolution at multiple temperatures in the \TR\ to map the 
vertical transport of energy over a wide \FOV.
In the latest configuration, the \MOSES\ optics are optimized around Ne\,\textsc{vii} (\SI{0.5}{\mega\kelvin}).
To achieve our goals, \ESIS\ should have a complementary wavelength choice such that we can observe a reasonably 
isolated emission line formed in the lower \TR."""
        ))

        with doc.create(pylatex.Subsection('Magnetic Reconnection Events')):
            doc.append(pylatex.NoEscape(
                r"""Magnetic reconnection describes the re-arrangement of the magnetic topology wherein magnetic energy 
is converted to kinetic energy resulting in the acceleration of plasma particles.
Reconnection is implicated in many dynamic, high energy solar events.
Solar flares are a well studied example (\eg\,\citet{Priest02} and the references therein), however we have little hope 
of pointing in the right place at the right time to observe a significant flare event in a rocket flight lasting only 
five minutes.
Instead, we will search for signatures of magnetic reconnection in \TR\ spectral lines.      
%label to track table 1 references
\phantomsection
\label{t1_2}
A particular signature of reconnection in the \TR\ is the explosive energy release by ubiquitous, small scale events.
These \EEs\ are characterized as spatially compact ($\approx$\SI{1.5}{\mega\meter} length~\citep{Dere94}) line 
broadenings on the order of \SI{100}{\kilo\meter\per\second}~\citep{Dere91}.
They are observed across a range of \TR\ emission lines that span temperatures of \SI{20000}{}--\SI{250000}{\kelvin} 
(C\,\textsc{ii}--O\,\textsc{v})~\citep{1994Moses}.
The typical lifetime of an \EE\ is 60-\SI{90}{\second}~\citep{1994Moses,Dere94,Dere91}.
Due to their location near quiet sun magnetic network elements, and the presence of supersonic flows near the Alfv\`en 
speed, \citet{Dere91} first suggested that \EEs\ may result from the model of fast Petschek~\citep{Petschek64} 
reconnection. 

The spectral line profile of \EEs\ may indicate the type of reconnection that is occurring in the \TR\ 
(\eg\,\citet{Rust17}).
For example, the Petschek model of reconnection predicts a `bi-directional jet' line profile with highly Doppler 
shifted wings, but little emission from the line core~\citep{Innes99}.
\citet{Innes15} developed a reconnection model resulting from a plasmoid instability~\citep{Bhattacharjee09}.
In contrast to the bi-directional jet, this modeled line profile has bright core emission and broad wings.
Both types of profile are seen in slit spectrograph data (\eg, \citet{Innes97,Innes15}, and the references therein), 
however \MOSES\ observed \EEs\ with more complicated morphologies than either of these two models 
suggest~\citep{Fox10,Rust17}.
It is unclear whether the differing observations are a function of wavelength and temperature, a result of a limited 
number of observations, or because the morphology of the event is difficult to ascertain from slit spectrograph data.

%label to track table 1 references
\phantomsection
\label{t1_01}
\ESIS\ will observe magnetic reconnection in the context of \EEs, by extending the technique pioneered by \MOSES\ to 
additional \TR\ lines.
Explosive events are well suited to sounding rocket observations;
a significant portion of their temporal evolution can be captured in $>$\SI{150}{\second} (\eg\,the analysis by 
\citet{Rust17}) and they are sufficiently common to provide a statistically meaningful sample in a 5-minute rocket 
flight (\eg,~\citet{Dere89,Dere91}).
In similarity with \MOSES, we seek a \TR\ line for \ESIS\ that is bright and well enough isolated from neighboring 
emission lines so as to be easily distinguished."""
            ))

        with doc.create(pylatex.Subsection('Energy Transfer')):
            doc.append(pylatex.NoEscape(
                r"""Tracking the mass and energy flow through the solar atmosphere is a long-standing goal in solar 
physics.
Bulk mass flow is evidenced by Doppler shifts or skewness in spectral lines.
However, the observed non-thermal broadening of \TR\ spectral lines may result from a variety of physical processes, 
including \MHD\ waves~\citep{DePontieu15, DePontieu07}, high-speed evaporative up-flows (\eg\,nanoflares, 
\citet{Patsourakos06}), turbulence, and other sources (\eg\,\citet{Mariska1992}).
This is a broad topic which \ESIS\ can address in many ways.
Here we will focus on a single application;
\ESIS\ will search for sources of Alfv\'en waves in the solar atmosphere by observing line broadening as the 
spectroscopic signature of these waves.

Alfv\'en waves in coronal holes are observed to carry an energy flux of 
\SI{7e5}{erg\per\centi\square\meter\per\second}, enough to energize the fast solar wind \citep{Hahn2012,Hahn2013}.
The source and frequency spectrum of these waves is unknown.
Here, we hypothesize that \MHD\ waves are similarly ubiquitous in quiet Sun and active regions, and play an important 
role in the energization of the quiescent corona.

%label to track table 1 references
\phantomsection
\label{t1_1}
The magnitude of non-thermal broadening of optically thin spectral lines is a direct measure of the wave 
amplitude~\citep{Banerjee09,Hahn2012,Hahn2013}.
We may estimate a lower limit on the non-thermal velocity to be observed as follows.
We assume that the magnetic field is constant for small changes in scale height in the \TR\ and that line of sight 
effects are negligible for observations sufficiently far from disk center.
Since the solar wind is not accelerated to an appreciable fraction of the Alfv\'en wave velocity at altitudes below 
$R \leq 1.15R_\odot$~\citep{Cranmer05}, the wave amplitude, $v_{nt}$, depends only weakly on electron density, $n_e$, so 
that $v_{nt} \propto n_e^{-1/4}$~\citep{Hahn2013,Moran01}.
Assuming pressure balance between the low corona and transition zone, we may infer non-thermal velocities in the \TR\ by 
scaling according to the temperature drop, $v_{nt} \propto T^{1/4}$.
The measured non-thermal velocity of \SI{24}{\kilo\meter\per\second} for Si\,\textsc{viii}~\citep{Doyle98} 
(\SI{0.8}{\mega\kelvin}~\citep{Moran03}) near the limb should, neglecting damping, correspond to velocities of at least 
\SI{21}{\kilo\meter\per\second} in mid \TR\ Ne\,\textsc{vii}, and \SI{18}{\kilo\meter\per\second} in the lower 
O\,\textsc{v} (\SI{0.25}{\mega\kelvin}) line.
The above non-thermal velocities are arrived at assuming both O\,\textsc{v} and Ne\,\textsc{vii} are formed near their 
ionization equilibrium temperatures.
For O\,\textsc{v}, the thermal width is $\sim$\SI{11}{\kilo\meter\per\s} at \SI{0.25}{\mega\kelvin} which means the 
total linewidth is primarily due to the non-thermal component.    
    
More recently, ~\citet{Srivastava17} observed torsional Alfv\'en waves with amplitude 
$\sim$\SI{20}{\kilo\meter\per\second} and period $\sim$\SI{30}{\second} in the chromosphere.
Modeling shows that these torsional waves can transfer a significant amount of energy to the corona~\citep{Kudoh99}.
The torsional motion will be observed as Doppler shifts when viewed from the side.
The oscillation period is long enough to be well resolved but short enough to see $\sim$\SI{10}{} cycles in a single 
rocket flight.
An \ESIS-like instrument is therefore well suited to observations of torsional Alfv\'en wave propagation over multiple 
heights in the \TR. 

By mapping Doppler velocities over a wide field of view in the \TR, \ESIS\ can address questions about both the origin 
of waves and whether they are able to propagate upward into the corona.
Independent of the two propagation modes discussed above, there is a range of possible sources for Alfv\'en 
(and other \MHD) waves in the solar atmosphere.
Three potential scenarios are: \begin{inparaenum}[(1)] \item Waves originate in the chromosphere or below and propagate 
through the \TR\ at a spatially uniform intensity; \label{wave-1}
\item Intense sources are localized in the \TR, but fill only a fraction of the surface\label{wave-2}; and \item Weak 
sources are localized in the \TR, but cover the surface densely enough to appear like the first case\label{wave-3}. 
\end{inparaenum}
The resulting non-thermal widths for localized sources will be significantly higher than the 
$\sim$\SI{20}{\kilo\meter\per\second} mean derived above.
The concentration of non-thermal energy observed by \ESIS\ will serve as an indicator of source density.
\roy{Should we remove this section considering that \MOSES\ didn't work?}
Comparison of Doppler maps from \ESIS\ and \MOSES\ will indicate whether a uniform source density originates in the 
chromosphere or below (scenario~\ref{wave-1}) or is associated with spatially distributed \TR\ phenomena 
(scenario~\ref{wave-3}) such as explosive events, or macrospicules.
Comparison with a wider selection of ground and space based imagery will allow us to determine whether intense, 
localized sources (scenario~\ref{wave-2}) are associated with converging or emerging magnetic bipoles, type \textsc{ii} 
spicules, spicule bushes, or other sources beneath the \TR.
For these comparisons, we need only to localize, rather than resolve, wave sources.
A spatial resolution of $\sim$\SI{2}{\mega\meter} will be sufficient to localize sources associated with magnetic flux 
tubes that are rooted in photospheric inter-granular network lanes (\eg\,\citet{Berger95ApJ})."""
            ))

            with doc.create(pylatex.Subsection('Science Requirements')):
                doc.append(pylatex.NoEscape(
                    r"""\ESIS\ will investigate two science targets; 
reconnection in explosive events, and the transport of mass and energy through the transition region.
The latter may take many forms, from \MHD\ waves of various modes to \EUV\ jets or macro-spicules.
To fulfill these goals, \ESIS\ will obtain simultaneous intensity, Doppler shift and line width images of the \OV\ line 
in the solar transition region at rapid cadence.
This is a lower \TR\ line (\SI{.25}{\mega\kelvin}) that complements the MOSES Ne\,\textsc{vii}.
The bright, optically thin \OVion\ emission line is well isolated except for the two coronal \MgXion\ lines.
These coronal lines can be viewed as contamination or as a bonus;
we expect that with the \numChannelsWords\ \ESIS\ projections it will be possible to separate the \OVion\ emission from 
that of \MgXion.
From the important temporal, spatial, and velocity scales referenced Sections~\ref{subsec:MagneticReconnectionEvents} 
and \ref{subsec:EnergyTransfer} we define the instrument requirements in Table~\ref{table:scireq} that are needed to 
meet our science goals."""
                ))

                with doc.create(kgpy.latex.FigureStar(position='htb!')) as figure:
                    figure.add_image(str(figures.bunch_pdf()), width=None)
                    figure.add_caption(pylatex.NoEscape(
                        r"""\roy{Plot of the \numEmissionLines\ brightest emission lines in the \ESIS\ passband.
Calculated using ChiantiPy, with the \chiantiAbundances\ abundances file, the \chiantiDEM\ \DEM\ file, and
$n_e T = $\,\chiantiPressure.}"""
                    ))
                    figure.append(kgpy.latex.Label('fig:bunch'))

                with doc.create(pylatex.Table(position='htb!')) as table:
                    table._star_latex_name = True
                    with table.create(pylatex.Center()) as centering:
                        with centering.create(pylatex.Tabular(table_spec='llll', )) as tabular:
                            tabular.escape = False
                            tabular.add_row(['Parameter', 'Requirement', 'Science Driver', 'Capabilities'])
                            tabular.add_hline()
                            tabular.add_row([
                                r'Spectral line',
                                r'\OVion\ \roy{\OV}',
                                r'Explosive events',
                                r'\OVion, \MgXion, \HeIion, Table~\ref{table:prescription}',
                            ])
                            tabular.add_row([
                                r'Spectral resolution',
                                r'\SI{18}{\kilo\meter\per\second} broadening \roy{\spectralResolutionRequirement}',
                                r'\MHD\ waves',
                                r'\dispersionDoppler, Table~\ref{table:prescription}',
                            ])
                            tabular.add_row([
                                r'Spatial resolution',
                                r'\SI{2}{\arcsecond} (\SI{1.5}{\mega\meter}) \roy{\angularResolutionRequirement (\spatialResolutionRequirement)}',
                                r'Explosive events',
                                r'\spatialResolution, Table~\ref{table:prescription}',
                            ])
                            tabular.add_row([
                                r'Desired \SNR',
                                r'\SI{17.3}{} \roy{\snrRequirement} in \CH',
                                r'\MHD\ waves in \CH',
                                r'$>$\SI{17.7}{} w/$20\times$\SI{10}{\second} exp., '
                                r'\S~\ref{subsec:SensitivityandCadence}',
                            ])
                            tabular.add_row([
                                r'Cadence',
                                r'\SI{15}{\second} \roy{\cadenceRequirement}',
                                r'Torsional waves',
                                r'\SI{10}{\second} eff., \S~\ref{subsec:SensitivityandCadence}',
                            ])
                            tabular.add_row([
                                r'Observing time',
                                r'$>$\SI{150}{\second} \roy{\observingTimeRequirement}',
                                r'Explosive events',
                                r'\SI{270}{\second}, \S~\ref{sec:MissionProfile}',
                            ])
                            tabular.add_row([
                                r'\FOV',
                                r'\SI{10}{\arcminute} \roy{\fovRequirement} diameter ',
                                r'Span \QS, \AR, and limb',
                                r'\fov, Table~\ref{table:prescription}',
                            ])
                        table.add_caption(pylatex.NoEscape(
                            r"""ESIS instrument requirements.  
AR is active region, QS quiet sun, and CH coronal hole."""
                        ))
                        table.append(kgpy.latex.Label('table:scireq'))

    with doc.create(pylatex.Section('The ESIS Instrument')):
        doc.append(pylatex.NoEscape(
            r"""\ESIS\ is a multiple projection slitless spectrograph that obtains line intensities, Doppler shifts, and 
widths in a single snapshot over a 2D \FOV.
Starting from the notional instrument described in Sec.~\ref{sec:TheESISConcept}, \ESIS\ has been designed to ensure all 
of the science requirements set forth in Table~\ref{table:scireq} are met.
The final design parameters are summarized in Table~\ref{table:prescription}.

A schematic diagram of a single \ESIS\ channel is presented in Fig.~\ref{fig:schematic}a, while the mechanical features 
of the primary mirror and gratings are detailed in Figs.~\ref{fig:schematic}b and \ref{fig:schematic}c, respectively."""
        ))

        with doc.create(pylatex.Table()) as table:
            table._star_latex_name = True
            table.escape = False
            with table.create(pylatex.Center()) as centering:
                with centering.create(pylatex.Tabular('lll')) as tabular:
                    tabular.escape = False
                    tabular.add_hline()
                    tabular.add_row([r'Primary', r'Surface shape', r'Parabolic'])
                    tabular.add_row([r'', r'Focal length ', r'\primaryFocalLength'])
                    tabular.add_row([r'', r'Aperture shape', r'Octagonal'])
                    tabular.add_row([r'', r'Aperture diameter', r'\primaryDiameter'])
                    tabular.add_row([r'', r'Coating', r'SiC \roy{\primaryCoatingMaterialShort} single layer, optimized for \OVwavelength'])
                    tabular.add_row([r'', r'\roy{Efficiency (\OV)}', r'\roy{\primaryEfficiency}'])

                    tabular.add_hline()
                    tabular.add_row([r'Field stop', r'Sky plane diameter', r'\fov'])
                    tabular.add_row([r'', r'Aperture shape', r'Octagonal'])
                    tabular.add_row([r'', r'Aperture diameter', r'\fieldStopDiameter'])

                    tabular.add_hline()
                    tabular.add_row([r'Gratings (\numChannels)', r'Surface shape', r'Spherical'])
                    tabular.add_row([r'', r'Surface radius', r'\gratingRadius'])
                    tabular.add_row([r'', r'Aperture shape', r'Trapezoidal'])
                    tabular.add_row([r'', r'Aperture height', r'\gratingHeight'])
                    tabular.add_row([r'', r'Aperture long base', r'\gratingLongWidth'])
                    tabular.add_row([r'', r'Aperture short base', r'\gratingShortWidth'])
                    tabular.add_row([r'', r'Ruling type', r'Varied line spacing'])
                    tabular.add_row([r'', r'Constant ruling spacing coefficient', r'\gratingRulingSpacing'])
                    tabular.add_row([r'', r'Linear ruling spacing coefficient', r'\gratingLinearRulingSpacingCoefficient'])
                    tabular.add_row([r'', r'Quadratic ruling spacing coefficient', r'\gratingQuadraticRulingSpacingCoefficient'])
                    tabular.add_row([r'', r'Input angle', r'\gratingInputAngle'])
                    tabular.add_row([r'', r'Output angle (\OV)', r'\gratingOutputAngle'])
                    tabular.add_row([r'', r'Manufacturing process', r'Individual master gratings'])
                    tabular.add_row([r'', r'Coating', r'Mg/Al/SiC \roy{\gratingCoatingMaterialShort} multilayer, optimized for \OVwavelength'])
                    tabular.add_row([r'', r'Efficiency \roy{(\OV)}', r'\SI{14}{\percent} \roy{\gratingEfficiency}'])
                    tabular.add_row([r'', r'Uncoated efficiency \roy{(\OV)}', r'\SI{39}{\percent} \roy{\gratingGrooveEfficiency}'])

                    tabular.add_hline()
                    tabular.add_row([r'Filters (\numChannels)', r'Aperture shape', r'Circular'])
                    tabular.add_row([r'', r'Aperture diameter', r'\filterDiameter'])
                    tabular.add_row([r'', r'Material', r'\filterMaterial'])
                    tabular.add_row([r'', r'Thickness', r'\filterThickness'])
                    tabular.add_row([r'', r'Mesh ratio', r'\filterMeshRatio'])
                    tabular.add_row([r'', r'Mesh material', r'\filterMeshMaterialShort'])
                    tabular.add_row([r'', r'\roy{Efficiency (\OV)}', r'\roy{\filterEfficiency}'])

                    tabular.add_hline()
                    tabular.add_row([r'Detectors (\numChannels)', r'Model', r'\detectorName'])
                    tabular.add_row([r'', r'Active area', r'\detectorPixelsX\ $\times$ \detectorPixelsY'])
                    tabular.add_row([r'', r'Pixel size', r'\detectorPixelSize'])
                    tabular.add_row([r'', r'Quantum efficiency \roy{(\OV)}', r'33\% \roy{\detectorQuantumEfficiency}'])
                    tabular.add_row([r'', r'Minumum cadence', r'\minCadence'])

                    tabular.add_hline()
                    tabular.add_row([r'System', r'Magnification', r'\magnification'])
                    tabular.add_row([r'', r'Plate scale', r'\plateScale'])
                    tabular.add_row([r'', r'Nyquist resolution', r'\spatialResolution'])
                    tabular.add_row([r'', r'Dispersion', r'\dispersion\ (\dispersionDoppler)'])
                    tabular.add_row([r'', r'Passband', r'\minWavelength\ to \maxWavelength'])
                    # tabular.add_row([r'Back focal length', r'\SI{127}{\milli\meter}'])

                    tabular.add_hline()

            table.add_caption(pylatex.NoEscape(r"""\ESIS\ design parameters."""))
            table.append(kgpy.latex.Label('table:prescription'))

        with doc.create(pylatex.Subsection('Optics')):
            doc.append(pylatex.NoEscape(
                r"""Figure~\ref{fig:schematic}a shows the relative layout of the optics and detectors for a single 
\ESIS\ channel.
Here we give specific details of the primary mirror and gratings (Fig.~\ref{fig:schematic}b and \ref{fig:schematic}c, respectively).
The features of the field stop have been described previously in Sec.~\ref{subsec:ESISFeatures}, while the \CCD\ and 
cameras are covered in Sec.~\ref{subsec:Cameras}. """
            ))

            with doc.create(kgpy.latex.FigureStar(position='htb!')) as figure:
                figure.append(kgpy.latex.aas.Gridline([
                    kgpy.latex.aas.Fig(figures.schematic_pdf(), kgpy.latex.textwidth, '(a)')
                ]))

                figure.append(kgpy.latex.aas.Gridline([
                    kgpy.latex.aas.LeftFig(figures.schematic_primary_and_obscuration_pdf(), kgpy.latex.columnwidth, '(b)'),
                    kgpy.latex.aas.RightFig(figures.schematic_grating_pdf(), kgpy.latex.columnwidth, '(c)'),
                ]))

                figure.add_caption(pylatex.NoEscape(
                    r"""(a) Schematic diagram of Channel 1 of the \ESIS\ optical system.
(b) Clear aperture of the primary mirror, size of the central obscuration, and the footprint of the beam for each 
channel.
(c) Clear aperture of Channel 1's diffraction grating."""
                ))
                figure.append(kgpy.latex.Label('fig:schematic'))

            doc.append(pylatex.NoEscape(
                r"""The primary mirror is octagonal in shape.
A triangular aperture mask in front of the primary mirror defines the clear aperture of each channel, imaged by a 
single grating.
Figure~\ref{fig:schematic}b shows one such aperture mask superimposed upon the primary mirror.
The octagonal shape of the primary also allows dynamic clearance for filter tubes that are arranged radially around the 
mirror (\S\,\ref{subsec:CoatingsandFilters}).
The mirror is attached to a backing plate by three ``bipods'';
thin titanium structures that are flexible in the radial dimension, perpendicular to the mirror edge, but rigid in the 
other two dimensions.
The bipods form a kinematic mount, isolating the primary mirror figure from mounting stress. 

The mirror will have to maintain its figure under direct solar illumination, so low expansion (Corning \ULE) substrates 
were used.
The transparency of \ULE, in conjunction with the transparency of the mirror coating in visible and near IR 
\roy{near-IR} wavelengths (\eg, Table~\ref{table:prescription} and \S\,\ref{subsec:CoatingsandFilters}), helps minimize 
the heating of the mirror.
Surface figure specifications for the \ESIS\ optics are described in Sec.~\ref{subsec:OptimizationandTolerancing}.

The spherical gratings (Fig.~\ref{fig:schematic}c) re-image light from the field stop to form dispersed images at the 
\CCDs.
Each grating is individually mounted to a backing plate in a similar fashion as the primary mirror.
For these much smaller optics, lightweight bipods were wire \EDM\ cut from thin titanium sheet.
The bipods are bonded to both the grating and backing plate along the three long edges of each grating.
The individual mounts allow each grating to be adjusted in tip and tilt to center the image on the \CCD. """
            ))

            with doc.create(pylatex.Table()) as table:
                with table.create(pylatex.Center()) as centering:
                    with centering.create(pylatex.Tabular('rr')) as tabular:
                        tabular.escape = False
                        tabular.add_row(['Channel', 'Serial number'])
                        tabular.add_hline()
                        for i in range(optics_all.channel_name.size):
                            index = np.unravel_index(i, optics_all.channel_name.shape)
                            tabular.add_row([optics_all.channel_name[index], optics_all.grating.serial_number[index]])

            with doc.create(pylatex.Figure()) as figure:
                # figure.add_image('figures/old/dispersion_opt1', width=pylatex.NoEscape('\columnwidth'))
                # figure.append('\n')
                figure.add_image(str(figures.field_stop_projections_pdf()), width=None)
                figure.add_caption(pylatex.NoEscape(
                    r"""Areas occupied by strong spectral lines on the \ESIS\ detectors.
The plot axes are sized exactly to the \CCD\ active area.
The \ESIS\ passband is defined by a combination of the field stop and grating dispersion."""
                ))
                figure.append(kgpy.latex.Label('fig:projections'))

            doc.append(pylatex.NoEscape(
                r"""The gratings have a varied line space ruling pattern optimized to provide, in principle, pixel 
limited \roy{pixel-limited} imaging from the field stop to the \CCDs.
The pitch at the center of the grating is $d_0=$\SI{.3866}{\micro\meter} \roy{\gratingRulingSpacing} resulting in a 
dispersion of \SI{17.5}{\kilo\meter\per\second} \roy{\dispersionDoppler} at the center of the \OV\ \FOV.
The groove profile is optimized for the $m=1$ order, so that each grating serves only a single \CCD.
The modeled grating groove efficiency in this order is \SI{36}{\percent} \roy{We said \SI{39}{\percent} above, need to 
find out which it is, I get \gratingGrooveEfficiency} at \SI{63}{\nano\meter} \roy{\OV}. 

Figure specification and groove profile are not well controlled near the edges of the gratings.
Therefore, a baffle is placed at each grating to restrict illumination to the clear aperture marked in 
Fig.~\ref{fig:schematic}c

The \ESIS\ passband is defined through a combination of the field stop, the grating dispersion, and the \CCD\ size.
The passband includes the He\,\textsc{i} (\SI{58.43}{\nano\meter}) \roy{\HeI} spectral line through Mg\,\textsc{x} 
(60.98 and \SI{62.49}{\nano\meter}) \roy{\MgXion\ (\MgXwavelength\ and \MgXdimWavelength)} to O\,\textsc{v} 
(\SI{62.97}{\nano\meter}) \roy{\OV}.
Figure~\ref{fig:projections} shows where images of each of the strong spectral lines will fall on the \CCD.
The instrument dispersion satisfies the spectral resolution requirement in Table~\ref{table:scireq} and ensures that the 
spectral images are well separated; Figure~\ref{fig:projections} shows that He\,\textsc{i} \roy{\HeI} will be completely 
separated from the target O\,\textsc{v} \roy{\OV} line."""
            ))

        with doc.create(pylatex.Subsection('Optimization and Tolerancing')):
            doc.append(pylatex.NoEscape(
                r"""The science resolution requirement of \angularResolutionRequirement (Table~\ref{table:scireq}) was 
flowed down to specifications for the \ESIS\ optics.
To ensure that \ESIS\ meets this requirement, an imaging error budget was developed to track parameters that 
significantly influence instrument resolution.
The budget is roughly divided into two categories;
the first includes `variable' parameters that can be directly controlled (\eg, the figure and finish of the optics, 
grating radius and ruling, placement of the elements in the system, and the accuracy to which the instrument is 
focused).
The second category consists of `fixed' contributions (\eg, \CCD\ charge diffusion, pointing stability, and diffraction 
from the entrance aperture).
In this sub-section we describe the optimization of the first category to balance the contributions of the second. 

Figure and surface roughness specifications for the primary mirror and gratings were developed first by a rule of thumb 
and then validated through a Fourier optics based model \roy{Fourier-optics-based model} and MonteCarlo \roy{MonteCarlo} simulations.
Surface figure errors were randomly generated, using a power law distribution in frequency.
The model explored a range of power spectral distributions for the surface figure errors, with power laws ranging from 
0.1 to 4.0.
For each randomly generated array of optical figure errors, the amplitude was adjusted to yield a target \MTF\ 
degradation factor, as compared to the diffraction limited \roy{diffraction-limited} \MTF.
For the primary mirror, the figure of merit was a \MTF\ degradation of 0.7 at \angularResolutionRequirement\ resolution.
Though the grating is smaller and closer to the focal plane, it was allocated somewhat more significant \MTF\ 
degradation of 0.6 based on manufacturing capabilities.
The derived requirements are described in table~\ref{table:error}.
Note that this modeling exercise was undertaken before the baffle designs were finalized.
The estimated diffraction \MTF\ and aberrations were therefore modeled for a rough estimate of the \ESIS\ single sector 
aperture."""
            ))

            with doc.create(pylatex.Table()) as table:
                # table._star_latex_name = True
                with table.create(pylatex.Center()) as centering:
                    with centering.create(pylatex.Tabular('llrr')) as tabular:
                        tabular.escape = False
                        tabular.add_row([r'Element', r'Parameter', r'Requirement', r'Measured'])
                        tabular.add_hline()
                        tabular.add_row([r'Primary', r'RMS slope error ($\mu$rad)', r'$<1.0$', r''])
                        tabular.add_row([r'', r'Integration length (mm)', r'4.0', r''])
                        tabular.add_row([r'', r'Sample length (mm)', r'2.0', r''])
                        tabular.add_hline()
                        tabular.add_row([r'Primary', r'RMS roughness (mm)', r'$<2.5$', r''])
                        tabular.add_row([r'', r'Periods (mm)', r'0.1-6', r''])
                        tabular.add_hline()
                        tabular.add_row([r'Grating', r'RMS slope error ($\mu$rad)', r'$<3.0$', r''])
                        tabular.add_row([r'', r'Integration length (mm)', r'2 \roy{why fewer sigfigs?}', r''])
                        tabular.add_row([r'', r'Sample length (mm)', r'1', r''])
                        tabular.add_hline()
                        tabular.add_row([r'Grating', r'RMS roughness (mm)', r'$<2.3$', r''])
                        tabular.add_row([r'', r'Periods (mm)', r'0.02-2', r''])
                        tabular.add_hline()
                table.add_caption(pylatex.NoEscape(
                    r"""Figure and surface roughness requirements compared to metrology for the \ESIS\ optics.
Slope error (both the numerical estimates and the measurements) is worked out with integration length and sample length 
defined per ISO 10110."""
                ))
                table.append(kgpy.latex.Label('table:error'))

            doc.append(pylatex.NoEscape(
                r"""The initial grating radius of curvature, $R_g$, and ruling pattern of the \ESIS\ gratings were 
derived from the analytical equations developed by \citet{Poletto04} for stigmatic spectrometers.
A second order polynomial describes the ruling pattern,
\begin{equation} \label{Eq-d}
    d = d_0 + d_1 r + d_2 r^2 \, ,
\end{equation}
where $r$ runs radially outward from the optical axis with its origin at the center of the grating \roy{shouldn't we be talking about $x$ here?}
(Fig.~\ref{fig:schematic}c).
The parameters of Equation~\ref{Eq-d} and $R_g$ were chosen so that the spatial and spectral focal curves intersect at 
the center of the O\,\textsc{v} \roy{\OV} image on the \CCD.

Starting from the analytically derived optical prescription, a model of the system was developed in ray-trace \roy{raytrace} software.
Since the instrument is radially symmetric, only one grating and its associated lightpath was analyzed. \roy{delete previous sentence, all lightpaths were analyzed}
In the ray trace model, $R_g$, $d_1$, $d_2$, grating cant angle, \CCD\ cant angle, and focus position were then 
optimized to minimize the RMS spot at select positions in the O\,\textsc{v} \roy{\OV} \FOV, illustrated in Fig.~\ref{fig:psf}.
The optical prescription derived from the ray trace is listed in Table~\ref{table:prescription} and 
Figure~\ref{fig:schematic}. """
            ))

            with doc.create(pylatex.Figure()) as figure:
                figure.add_image(str(figures.psf_pdf()), width=None)
                figure.add_caption(pylatex.NoEscape(
                    r"""
\roy{
Raytraced spot diagrams for \OV\ with $\psfFieldSamples \times \psfFieldSamples$ field angles across the \FOV.
The box around each spot represents a single pixel on the detector.
Each spot was traced using a stratified random grid across the pupil with $\psfPupilSamples \times \psfPupilSamples$ 
positions per spot.
}
(Left:)  Ray traced spot diagrams for \ESIS, illustrated at the center and vertices of the O\,\textsc{v} FOV on the 
\CCD.
The grid spacing is \SI{1}{\micro\meter} and the diffraction limit airy disk (overplotted on each spot) radius is \SI{2}{\micro\meter}.
Imaging performance will be limited by the \SI{15}{\micro\meter} pixel size.
(Right:) RMS spot radius through focus for the three centered spots; top of FOV (purple curve), center (maroon), and bottom (red)."""
                ))
                figure.append(kgpy.latex.Label('fig:psf'))

            with doc.create(kgpy.latex.FigureStar()) as figure:
                figure.add_image(str(figures.spot_size_pdf()), width=None)
                figure.add_caption(pylatex.NoEscape(
                    r"""\roy{
2D histogram of RMS spot sizes for the \defaultNumEmissionLines\ brightest wavelengths in the \ESIS\ passband. 
Each wavelength has $\defaultFieldSamples \times \defaultFieldSamples$ field points across the \FOV, and each field point
has a stratified random grid containing $\defaultPupilSamples \times \defaultPupilSamples$ pupil positions.
The \HeI\ line appears cropped since it is cut off by the edge of the detector.
The images appear flipped compared to Figure~\ref{fig:projections} since the optical system inverts the image of the skyplane.
}"""
                ))
                figure.append(kgpy.latex.Label('fig:spotSize'))

            with doc.create(pylatex.Figure()) as figure:
                figure.add_image(str(figures.focus_curve_pdf()), width=None)
                figure.add_caption(pylatex.NoEscape(
                    r"""\roy{
Focus curve for the field angle at the middle of the \ESIS\ \FOV\ for the 
\defaultNumEmissionLines\ brightest wavelengths in the passband.
}"""
                ))
                figure.append(kgpy.latex.Label('fig:focusCurve'))

            with doc.create(pylatex.Figure()) as figure:
                figure.add_image(str(figures.vignetting_pdf()), width=None)
                figure.add_caption(pylatex.NoEscape(
                    r"""\roy{
(Top) 2D histogram counting the number of rays that were unvignetted by the \ESIS\ optical 
system as a function of field position.
The count is normalized to the maximum number of unvignetted rays at any field point.
The field and pupil grids have the same parameters as the grid for Figure~\ref{fig:spotSize}.
(Bottom) Residual between the top histogram and the vignetting model described in Table~\ref{table:vignetting}
}"""
                ))
                figure.append(kgpy.latex.Label('fig:vignetting'))

            with doc.create(pylatex.Figure()) as figure:
                figure.add_image(str(figures.distortion_pdf()), width=None)
                figure.add_caption(pylatex.NoEscape(
                    r"""\roy{
Plot of the magnified, undistorted field stop aperture vs. the distorted \OV\ image of the 
field stop aperture on the \ESIS\ detector.
The magnification factor used for the undistorted field stop aperture is the ratio of the grating exit arm to the 
grating entrance arm (\armRatio).
The distorted image of the field stop aperture was calculated using the \ESIS\ distortion model, described in 
Table~\ref{table:distortion}.
}"""
                ))
                figure.append(kgpy.latex.Label('fig:distortion'))

            with doc.create(pylatex.Figure()) as figure:
                figure._star_latex_name = True
                figure.add_image(str(figures.distortion_residual_pdf()), width=None)
                figure.add_caption(pylatex.NoEscape(
                    r"""\roy{
Magnitude of the residual between a linear distortion model and the raytrace model (top) and between a quadratic 
distortion model and the raytrace model (bottom). This figure demonstrates that a quadratic distortion model is
sufficient to achieve sub-pixel accuracy.
}"""
                ))
                figure.append(kgpy.latex.Label('fig:distortionResidual'))

            doc.append(pylatex.NoEscape(
                r"""The ray trace model was also used to quantify how mirror and positional tolerances affect the 
instrument's spatial resolution.
Each element of the model was individually perturbed, then a compensation applied to adjust the image on the \CCD.
The compensation optimized grating tip/tilt angle and \CCD\ focus position, so that the image was re-centered and RMS 
spot size minimized at the positions in Fig.~\ref{F-spot} \roy{minimized at the vertices of the field stop and the central field angle}.
We then computed the maximum change in RMS spot size over all spot positions between the optimized and perturbed models.
The computed positional tolerances for each element in the \ESIS\ optical system are listed in Table~\ref{table:tol}.

The imaging error budget is displayed in Table~\ref{table:tol}.
For the primary mirror and grating surface figure contributions, we choose the \MTF\ figures of merit from the surface 
roughness specifications described earlier.
To quantify the remaining entries, we assume that each term can be represented by a gaussian function of width 
$\sigma^2$ that ``blurs'' the final image.
The value of $\sigma$ then corresponds to the maximum change in RMS spot size for each term as it is perturbed in the 
tolerance analysis described above.
The value of the \MTF\ in the right-most column of Table~\ref{table:tol} is computed from 
each of the gaussian blur terms at the Nyquist frequency (\SI{0.5}{cycles\per arcsecond}).
From Table~\ref{table:tol}, we estimate the total \MTF\ of ESIS to be $0.109$ at the Nyquist frequency.
Compared to, for example, the Rayleigh criterion of \SI{0.09}{cycles\per arcsecond}~\citep{Rayleigh_1879} we estimate 
the resolution of \ESIS\ to be essentially pixel limited.
Since \ESIS\ pixels span \SI{0.76}{\arcsecond}, the resolution target in Table~\ref{table:scireq} is obtained by this 
design."""
            ))

            doc.append(pylatex.NoEscape(
                r"""
\begin{table*}[!htb]
\caption{Imaging error budget and tolerance analysis results.  \MTF\ is given at \protect\SI{0.5}{cycles\per arcsecond}.}
\begin{tabular}{llrcc}
Element 	&	  		& Tolerance & $\sigma^2$ $[\mu m]$ & \MTF \\
\hline %-----------------------------------------------------------------------------
Primary M.	& Surface figure & (Ref. Table~\ref{table:surfaces}) & & 0.700 \\
			& Decenter	& 1 \si{\milli\meter}		& 1.700 	& 0.881 \\
            %& Defocus	& .08			& 3.9 		&  \\
Grating		& Surface figure & (Ref. Table~\ref{table:surfaces}) & & 0.600 \\
			& Radius	& 2.5 \si{\milli\meter}		& 1.410		& 0.916 \\
            & Decenter	& 1 \si{\milli\meter}		& 0.001		& 1.000 \\
            & Defocus	& 0.015 \si{\milli\meter}	& 0.801		& 0.972 \\
            & Clocking  & 13 \si{\milli\radian} 	& 1.300 	& 0.929 \\
CCD			& Decenter	& 1 \si{\milli\meter}		& 0.310		& 0.996 \\
			& Defocus 	& 0.229 \si{\milli\meter}	& 0.706 	& 0.978 \\            
\hline %-----------------------------------------------------------------------------
\multicolumn{2}{l}{Max RMS spot radius (modeled)} 	&				& 1.720			& 0.878 \\
\multicolumn{2}{l}{CCD charge diffusion (est.)} &				& 2.000			& 0.839 \\
Thermal drift	& 		&				& 0.192			& 0.998 \\
SPARCS drift	& 		&				& 1.920			& 0.998 \\
Pointing jitter & 		&				& 3.430			& 0.597 \\
Diff. Limit 	& 		&				&				& 0.833 \\
\hline %-----------------------------------------------------------------------------
Total \MTF\	 	& 		&				&				& 0.109 \\
\end{tabular}
\label{table:tol}
\end{table*}
"""
            ))
            # with doc.create(pylatex.Table()) as table:
            #     # table._star_latex_name = True
            #     with table.create(pylatex.Center()) as centering:
            #         with centering.create(pylatex.Tabular('llrcc')) as tabular:

        with doc.create(pylatex.Subsection('Coatings and Filters')):

            with doc.create(pylatex.Figure()) as figure:
                figure.add_image(str(figures.grating_multilayer_schematic_pdf()), width=None)
                figure.add_caption(pylatex.NoEscape(
                    r"""
Schematic of the Al/SiC/Mg \roy{\gratingCoatingMaterialShort} multilayer with $N=4$ \roy{$N=\gratingCoatingNumLayers$} layers.
"""
                ))
                figure.append(kgpy.latex.Label('fig:gratingMultilayerSchematic'))

            with doc.create(pylatex.Figure()) as figure:
                figure.add_image(str(figures.grating_efficiency_vs_angle_pdf()), width=None)
                figure.add_caption(pylatex.NoEscape(
                    r"""
Measured efficiency \roy{at \gratingTestWavelength} of a single grating \roy{the Channel \testGratingChannelIndex\ grating} as a function of reflection angle on \roy{\testGratingDate}.
Note flat response in first order over instrument \FOV\ and suppression of zero order.
"""
                ))
                figure.append(kgpy.latex.Label('fig:gratingEfficiencyVsAngle'))

            with doc.create(pylatex.Figure()) as figure:
                figure.add_image(str(figures.component_efficiency_vs_wavelength_pdf()), width=None)
                figure.add_caption(pylatex.NoEscape(
                    r"""(Top) Measured reflectance for several multilayer coated witness samples 
\roy{at an incidence angle of \gratingWitnessMeasurementIncidenceAngle\ on \testGratingDate.
Note the suppression of second order relative to the first order.
(Bottom) Comparison of the efficiency of the three main \ESIS\ optical components: primary mirror, grating and filter.
The primary mirror efficiency is based on measurements of a \Si\ witness sample taken on \primaryMeasurementDate\ at an 
angle of incidence of \primaryWitnessMeasurementIncidenceAngle. 
The grating efficiency is from a measurement of the Channel \testGratingChannelIndex\ grating taken on \testGratingDate\
at an angle of incidence of \gratingMeasurementIncidenceAngle.
The filter efficiency is a theoretical model that includes the filter mesh, \filterThickness\ of \filterMaterial\ and
\filterOxideThickness\ of \filterMaterial\ oxide.
}"""
                ))
                figure.append(kgpy.latex.Label('fig:componentEfficiencyVsWavelength'))

            with doc.create(pylatex.Figure()) as figure:
                figure._star_latex_name = True
                figure.add_image(str(figures.grating_efficiency_vs_position_pdf()), width=None)
                figure.add_caption(pylatex.NoEscape(
                    r"""
\roy{Channel \testGratingChannelIndex\ grating efficiency at \gratingTestWavelength\ vs. position for two orthogonal slices across the optical 
surface on \testGratingDate.}"""
                ))
                figure.append(kgpy.latex.Label('fig:gratingEfficiencyVsPosition'))

            doc.append(pylatex.NoEscape(
                r"""
The diffraction gratings are coated with a multilayer optimized for a center wavelength of \SI{63.0}{\nano\meter} \roy{\OV}, 
developed by a collaboration between Reflective X-Ray Optics LLC and \LBNL.
In Fig.~\ref{fig:gratingEfficiencyVsAngle}, characterization of a single, randomly selected multilayer coated grating at \LBNL\ shows 
that the grating reflectivity is constant over the instrument \FOV\ in the $m=1$ order while the $m=0$ order is almost 
completely suppressed.
Figure~\ref{fig:gratingMultilayerSchematic} shows a schematic of the coating that achieves peak reflectivity and selectivity in the 
$m=0$ order using four \roy{\gratingCoatingNumLayersWords} layer pairs of silicon carbide (SiC) \roy{\firstGratingCoatingMaterial} and magnesium (Mg) \roy{\secondGratingCoatingMaterial}.
The Aluminum (Al) \roy{\thirdGratingCoatingMaterial} layers are deposited adjacent to each Mg \roy{\firstGratingCoatingMaterial} layer to mitigate corrosion.
\roy{As Charles mentioned, this doesn't make sense. Why can it go \firstGratingCoatingMaterial\ to \secondGratingCoatingMaterial, but not \secondGratingCoatingMaterial\ to \firstGratingCoatingMaterial?}

The maximum reflectance for the coating alone in the nominal instrument passband is $\sim$\SI{35}{\percent} \roy{\gratingWitnessEfficiency} in 
the upper panel of Figure~\ref{fig:componentEfficiencyVsWavelength}, measured from witness samples coated at the same time as the diffraction gratings.
Combined with the predicted groove efficiency from \S\,\ref{subsec:Optics} and, given the relatively shallow groove profile 
and near normal incidence angle, the total reflectivity in first order is $\sim$\SI{13}{\percent} \roy{\gratingEfficiency} at 
\SI{63}{\nano\meter} \roy{\OV}.
This is confirmed by the first order efficiency measured from a single \ESIS\ grating in the lower panel of 
Figure~\ref{fig:componentEfficiencyVsWavelength}.  

Unlike \EUV\ imagers (\eg, \TRACE~\citep{Handy99}, \AIA~\citep{Lemen12}, and Hi-C~\citep{Kobayashi2014}) 
the \ESIS\ passband is defined by a combination of the field stop and grating (\S\,\ref{subsec:Optics}, 
Fig.~\ref{fig:projections}) rather than multi-layer coatings.
The coating selectivity is therefore not critical in this respect, allowing the multi-layer to be manipulated to 
suppress out-of-band bright, nearby emission lines.
The lower panel of Figure~\ref{fig:componentEfficiencyVsWavelength} shows the peak reflectance of the grating multilayer is shifted slightly towards longer 
wavelengths to attenuate the He\,\textsc{i} emission line, reducing the likelihood of detector saturation.
A similar issue arises with the bright He\,\textsc{ii} (\SI{30.4}{\nano\meter}) line.
Through careful design of the grating multilayer, the reflectivity at this wavelength is $\sim$\SI{2}{\percent} of that 
at \SI{63}{\nano\meter} \roy{\OV} (lower panel of Figure~\ref{fig:componentEfficiencyVsWavelength}).
In combination with the primary mirror coating (described below) the rejection ratio at \SI{30.4}{\nano\meter} is 
$\sim$\SI{32}{\decibel}.  Thus, He\,\textsc{ii} emission will be completely attenuated at the \CCD.

The flight and spare primary mirrors were coated with the same Al/SiC/Mg multilayer.
Corrosion of this multilayer rendered both mirrors unusable.
The failed coating was stripped from primary mirror SN001.
The mirror was then re-coated with a \SI{5}{\nano\meter} \roy{\primaryCoatingBaseThickness} thick layer of chromium (Cr) \roy{\primaryCoatingBaseMaterial} to improve adhesion followed by a 
\SI{25}{\nano\meter} \roy{\primaryCoatingThickness} thick layer of SiC \roy{\primaryCoatingMaterial}.
The reflectance of this coating deposited on a \Si\ wafer witness sample appears in 
Fig.~\ref{fig:componentEfficiencyVsWavelength}.
The spare primary mirror (SN002) retains the corroded Al/SiC/Mg multilayer.

The \Si\ \CCDs\ are sensitive to visible light as well as \EUV.
Visible solar radiation is much stronger than \EUV, and visible stray light can survive multiple scatterings while 
retaining enough intensity to contaminate the \EUV\ images.
Lux\'el \citep{Powell90} Al \roy{\filterMaterial} filters \SI{100}{\nano\meter} \roy{\filterThickness} thick will be 
used to shield each \CCD\ from visible light.
The Al film is supported by a 70 line per inch (lpi) Ni \roy{\filterMeshMaterial} mesh, with 82\% \roy{\filterMeshRatio} transmission.
The theoretical filter transmission curve, modeled from CXRO data \citep{Henke93}, is displayed in 
Fig.~\ref{fig:componentEfficiencyVsWavelength}.
We conservatively estimate filter oxidation at the time of launch as a 4nm thick layer of Al$_2$O$_3$.

An Al filter is positioned in front of the focal plane of each \CCD\ by a filter tube, creating a light tight box with a 
labyrinthine evacuation vent (e.g., Fig.~\ref{F-cameras}).
The placement of the filter relative to the \CCD\ is optimized so that the filter mesh shadow is not visible.
By modeling the filter mesh shadow, we find that a position far from the CCD ($>$\SI{200}{\milli\meter}) and mesh grid
clocking of \SI{45}{\degree} to the detector array reduces the shadow amplitude well below photon statistics.
The \MOSES\ instrument utilizes a similar design; no detectable signature of the filter mesh is found in data and 
inversion residuals from the 2006 \MOSES\ flight.

To prevent oxidation, and to minimize the risk of tears, pinholes, and breakage from handling, the filters will be 
stored in a nitrogen purged environment until after payload vibration testing."""
            ))

        with doc.create(pylatex.Subsection('Sensitivity and Cadence')):
            doc.append(pylatex.NoEscape(
                r"""
Count rates for ESIS are estimated using the expected component throughput from Section~\ref{subsec:CoatingsandFilters} and the CCD quantum efficiency (QE) listed in Table~\ref{table:prescription}.
Line intensities are derived from \citet{Vernazza78} (V\&R) and the SOHO/\CDS\ \citep{Harrison95} data.
The \SI{100}{\percent} duty cycle of ESIS (\S\,\ref{subsec:Cameras}) gives us the flexibility to use the shortest exposures that are scientifically useful.
So long as the shot noise dominates over read noise (which is true even for our coronal hole estimates at \SI{10}{\second} exposure length), we can stack exposures without a significant SNR penalty.
Table~\ref{table:count} shows that ESIS is effectively shot noise limited with a \SI{10}{\second} exposure.
The signal requirement in Table~\ref{table:scireq} is met by stacking exposures.
Good quality images ($\sim300$ counts) in active regions can be obtained by stacking \SI{30}{\second} worth of exposures.
This cadence is sufficient to observe explosive events, but will not resolve torsional Alfv\'en waves described in \S\,\ref{sec:ScienceObjectives}.
However, by stacking multiple \SI{10}{\second} exposures, sufficient SNR \emph{and} temporal resolution of torsional Alfv\'en wave oscillations can be obtained.
We also note that the count rates given here are for an unvignetted system which is limited by the baffling of this design.
While not explored here, there is the possibility of modifying the instrument baffling (\S\,\ref{subsec:AperturesandBaffles}) to increase throughput.
Thus, a faster exposure cadence may be obtained by accepting some vignetting in the system.

\begin{table}
    \centering
    \begin{tabular}{lcccc}
        Source & V\&R & V\&R & V\&R & CDS \\
        Solar Context & QS & CH & AR & AR \\
        \hline
        \multicolumn{5}{c}{\SI{10}{\second} Exp.}\\
        Mg\,\textsc{x} (\SI{62.5}{\nano\meter}) & 3 & 0 & 26 & 16  \\
        O\,\textsc{V} (\SI{62.9}{\nano\meter}) & 22 & 19 & 66 & 34 \\
        \hline
        Total Counts & 25 & 19 & 92 & 50 \\
        Shot Noise & 5.0 & 4.3 & 9.6 & 7.0 \\
        Read Noise (est.) & \multicolumn{4}{c}{-- 1.9 --} \\
        SNR & 4.7 & 4.0 & 9.4 & 6.8 \\
        \hline \hline
        \multicolumn{5}{c}{$3\times$\SI{10}{\second} Exp. Stack}\\
        Total Counts & 75 & 56 & 276 & 148 \\
        SNR & 8.1 & 6.8 & 16.3 & 11.7 \\
        \hline
    \end{tabular}
    \caption{
        Estimated signal statistics per channel (in photon counts) for ESIS lines in coronal hole (CH), quiet Sun (QS), and active region (AR).
    }
    \label{table:count}
\end{table}
"""
            ))

        intensity_o5 = [334.97, 285.77, 1018.65, 519.534] * u.erg / u.cm ** 2 / u.sr / u.s
        intensity_mg10 = [51.43, 2.62, 397.64, 239.249] * u.erg / u.cm ** 2 / u.sr / u.s

        energy_o5 = wavelength_o5.to(u.erg, equivalencies=u.spectral()) / u.photon
        energy_mg10 = wavelength_mg10_2.to(u.erg, equivalencies=u.spectral()) / u.photon

        optics_single_measured = optics.as_measured_single_channel()
        rays = optics_single_measured.rays_output

        area = rays.intensity.copy()
        area[~rays.mask] = np.nan
        area = np.nansum(area, (rays.axis.pupil_x, rays.axis.pupil_y, rays.axis.velocity_los), keepdims=True)
        area[area == 0] = np.nan
        area = np.nanmean(area, (rays.axis.field_x, rays.axis.field_y)).squeeze()
        area_o5 = area[0]
        area_mg10 = area[2]

        pixel_subtent = (optics_single.plate_scale.x * optics_single.plate_scale.y * u.pix * u.pix).to(u.sr)
        time_integration = 10 * u.s

        counts_o5 = (intensity_o5 * area_o5 * pixel_subtent * time_integration / energy_o5).to(u.photon)
        counts_mg10 = (intensity_mg10 * area_mg10 * pixel_subtent * time_integration / energy_mg10).to(u.photon)
        counts_total = counts_o5 + counts_mg10

        stack_num = 3
        counts_total_stacked = counts_total * stack_num

        noise_shot = np.sqrt(counts_total.value) * counts_total.unit
        noise_shot_stacked = np.sqrt(counts_total_stacked.value) * counts_total.unit

        noise_read = optics_single_measured.detector.readout_noise.mean()
        noise_read = noise_read * optics_single_measured.detector.gain.mean()
        noise_read_o5 = (noise_read / (energy_o5 / (3.6 * u.eV / u.electron))).to(u.photon)
        noise_read_o5_stacked = stack_num * noise_read_o5

        noise_total = np.sqrt(np.square(noise_shot) + np.square(noise_read_o5))
        noise_total_stacked = np.sqrt(np.square(noise_shot_stacked) + np.square(noise_read_o5_stacked))

        snr = counts_total / noise_total
        snr_stacked = counts_total_stacked / noise_total_stacked

        label = f'1 $\\times$ {kgpy.format.quantity(time_integration, digits_after_decimal=0)} exp.'
        label_stacked = f'{stack_num} $\\times$ {kgpy.format.quantity(time_integration, digits_after_decimal=0)} exp.'

        with doc.create(pylatex.Table()) as table:
            # table._star_latex_name = True
            with table.create(pylatex.Center()) as centering:
                with centering.create(pylatex.Tabular('llrrrr')) as tabular:
                    tabular.escape = False
                    tabular.add_row([r'Source', r'', r'V\&R', r'V\&R', r'V\&R', r'\CDS'])
                    tabular.add_row(r'Solar context', r'', r'\QS', r'\CH', r'\AR', r'\AR')
                    tabular.add_hline()
                    tabular.add_row([label, r'\OV', ] + [f'{c:0.0f}' for c in counts_o5.value])
                    tabular.add_row([r'', r'\MgXdim',] + [f'{c:0.0f}' for c in counts_mg10.value])
                    tabular.add_hline()
                    tabular.add_row([r'', r'Total', ] + [f'{c:0.0f}' for c in counts_total.value])
                    tabular.add_row([r'', r'Shot noise', ] + [f'{c:0.1f}' for c in noise_shot.value])
                    tabular.add_row([r'', r'Read noise', ] + 4 * [f'{noise_read_o5.value:0.1f}'])
                    tabular.add_row([r'', r'\SNR', ] + [f'{c:0.1f}' for c in snr.value])
                    tabular.add_hline()
                    tabular.add_hline()
                    tabular.add_row([label_stacked, 'Total', ] + [f'{c:0.0f}' for c in counts_total_stacked.value])
                    tabular.add_row([r'', r'\SNR', ] + [f'{c:0.1f}' for c in snr_stacked.value])

            table.add_caption(pylatex.NoEscape(
                r"""
Estimated signal statistics per channel (in photon counts) for \ESIS\ lines in \CH, \QS, and \AR."""
            ))
            table.append(kgpy.latex.Label('table:counts'))

        with doc.create(pylatex.Subsection('Alignment and Focus')):
            doc.append(pylatex.NoEscape(
                r"""
In the conceptual phase of ESIS, the decision was made to perform focus and alignment in visible light with a HeNe 
source.
Certain difficulties are introduced by this choice, however, the benefits outweigh the operational complexity and 
equipment that would be required for focus in EUV.
Moreover, a sounding rocket instrument requires robust, adjustment-free mounts to survive the launch environment.
Such a design is not amenable to iterative adjustment in vacuum.  The choice of alignment wavelength is arbitrary for 
most components;
CCD response and multilayer coating reflectively is sufficient across a wide band a visible wavelengths.
The exceptions are the thin film filters (which will not be installed until just before launch and have no affect on 
telescope alignment and focus) and the diffraction gratings.
Visible light gratings have been manufactured specifically for alignment and focus.
These gratings are identical to the EUV flight version, but with a ruling pattern scaled to a 
\SI{632.8}{\nano\meter} wavelength.
"""
            ))

            with doc.create(pylatex.Figure()) as figure:
                figure.add_image('figures/old/Alignment_transfer_1gr_text', width=kgpy.latex.columnwidth)
                figure.add_caption(pylatex.NoEscape(
                    r"""
ESIS alignment transfer device, consisting of three miniature confocal microscopes that translate along the optical 
axis.  
Trapezoidal grating, bipods, and mounting plate are installed on the tuffet in front of the apparatus 
(left of center)"""
                ))
                figure.append(kgpy.latex.Label('F-alt'))

            doc.append(pylatex.NoEscape(
                r"""
After alignment and focus has been obtained with the HeNe source, the instrument will be prepared for flight by 
replacing the visible gratings by the EUV flight versions.
Each grating (EUV and alignment) is individually mounted to a backing plate using a bipod system similar to that of the 
primary mirror.
The array of gratings on their backing plates are in turn mounted to a single part which we call the `tuffet.'
The backing plate is attached to the tuffet by bolts through spherical washers.
With this mounting scheme, the gratings can be individually aligned in tip/tilt.
The tuffet is attached to the secondary mirror mount structure (Fig.~\ref{F-alt}).
This enables the entire grating array to be replaced simply by removing/installing the tuffet, such as when switching 
between alignment and EUV gratings.

Properly positioning the gratings will be the most difficult aspect of final telescope assembly.
Table~\ref{table:tol} shows that the telescope is very sensitive to defocus.
The depth of field is an order of magnitude smaller in EUV than in visible light.
Moreover, the sensitivity to defocus at the gratings is $M^2+1=17$ times greater than at the detectors.
Another sensitive aspect of the telescope is grating tip/tilt.
A tolerance of $\sim\pm$\SI{0.5}{\milli\radian} will be needed to insure that the entire image lands on the active area 
of the CCD.

Once the visible gratings are aligned and focused, the challenge is to transfer this alignment to the UV gratings.
\citet{Johnson18} describes a procedure and the apparatus constructed to accurately transfer the position of an 
alignment grating radius to an EUV flight grating.
This device, displayed in Fig.~\ref{F-alt}, consists of an array of three miniature confocal microscopes that record the 
position of the alignment grating radius.
The alignment grating is replaced by an EUV grating, which is then measured into position by the same apparatus.
This device (and procedure) is capable of obtaining position measurements of the diffraction gratings to a repeatability 
of $\approx$\SI{14}{\micro\meter} in the three confocal channels.
This alignment transfer apparatus will ensure that the EUV flight gratings are positioned to within the tolerances 
described in Table~\ref{table:tol}."""
            ))

        with doc.create(pylatex.Subsection('Apertures and Baffles')):

            with doc.create(pylatex.Figure()) as figure:
                figure._star_latex_name = True
                figure.add_image('figures/old/Baffles_1clr', width=kgpy.latex.textwidth)
                figure.add_caption(pylatex.NoEscape(
                    r"""Model view of ESIS baffle placement and cutouts."""
                ))
                figure.append(kgpy.latex.Label('F-Baff1'))

            doc.append(pylatex.NoEscape(
                r"""
Each channel of ESIS has two apertures: one at the surface of the grating and another in front of the primary mirror.
The purpose of the aperture at the grating is to mask the out-of-figure margins at the edges of these optics.
This provides a well defined edge to the clear aperture of each grating while also keeping unwanted rays from being 
reflected from the grating margins and back onto the CCDs.
The dimensions of the grating aperture match those of the grating clear aperture shown in Figure~\ref{fig:schematic}c. 

The aperture placed at the primary mirror is the stop for each individual channel.
The area of the stop has been maximized under the constraint that no rays be vignetted anywhere else in the system.
The gratings and their clear apertures were the most significant areas of concern for potential vignetting.
Thus, the shape of stop at the primary is largely influenced by the shape of the grating clear aperture.
The inner extent of the primary stop (the ``tip'' of the triangle in Figure~\ref{fig:schematic}b) is defined by the 
occultation of the primary by the shadow cast from the gratings and their mounts.
This presented an intricate geometry problem, as the occultation is a function of the incoming field angle, 
the radial extent of the grating mount, and the distance of the mount to the primary mirror along the optical axis.
Hence, the inner extent of the primary stop was solved for iteratively with the optimization described in 
Section~\ref{subsec:OptimizationandTolerancing}, which affected the placement of the gratings relative to the primary mirror.
The resulting optimized and non-vignetting stop geometry is shown in Figure~\ref{fig:schematic}b.

After final optimization, the stop geometry was analyzed to check for vignetting at the grating with the optical model.
A footprint diagram was generated at the grating from of multiple grids of rays.
The incidence angle of each grid of rays corresponded to the extremes of FOV defined by the positions of the eight 
points of the octagonal field stop.
The footprint diagram showed that, with the stop completely filled, no ray landed outside of the grating clear aperture
in Figure~\ref{fig:schematic}c, and no ray was intercepted by the central obscuration.

From Figure~\ref{fig:schematic}c it is apparent that considerable surface area of the primary mirror is unused by the 
non-vignetting stop design.
The primary apertures could be enlarged considerably if the vignetting constraint were to be relaxed.

The ESIS baffles are designed to block direct light paths between the front aperture plate and the CCDs for any ray 
$<$\SI{1.4}{\degree} from the optical axis.
This angle is purposefully larger than the angular diameter of the sun ($\sim$\SI{0.5}{\degree}) so that any direct 
paths are excluded from bright sources in the solar corona.
All baffles are bead-blasted, anodized Al sheet metal oriented perpendicular to the optical axis.
The size and shape of the cutouts were determined using a combination of the ray trace from 
Section~\ref{subsec:OptimizationandTolerancing} and 3D modeling.
The light path from the primary mirror to the field stop is defined as the volume that connects each vertex of the 
primary mirror aperture mask (e.g., Fig.~\ref{fig:schematic}) to every vertex of the octagonal field stop.
This is a conservative definition that ensures no rays within the FOV are excluded, and therefore unintentionally 
vignetted by the baffles.  Light paths from the field stop to the grating, and from the grating to the image formed on 
the CCD, are defined in a similar manner.
The cutouts in the baffles are sized using the projection of these light paths onto the baffle surface.
A conservative \SI{1}{\milli\meter} margin is added to each cutout to prevent unintentional vignetting.
A model of the six baffles, showing cutouts and position on the optical bench, is displayed in Fig.~\ref{F-Baff1}."""
            ))

        with doc.create(pylatex.Subsection('Cameras')):

            with doc.create(pylatex.Figure()) as figure:
                figure.add_image('figures/old/ESIS_Cameras_1gr_text', width=kgpy.latex.columnwidth)
                figure.add_caption(pylatex.NoEscape(
                    r"""
ESIS camera assembly as built by MSFC.  
Thin film filters and filter tubes are not installed in this image."""
                ))
                figure.append(kgpy.latex.Label('F-cameras'))

            doc.append(pylatex.NoEscape(
                r"""
The ESIS CCD cameras were designed and constructed by Marshall Space Flight Center (MSFC), and are the latest in a 
series of camera systems developed specifically for use on solar space flight instruments.
The ESIS camera heritage includes those flown on both the Chromospheric Lyman-Alpha Spectro-Polarimeter 
(CLASP)~\citep{Kano12,Kobayashi12} and the High-Resolution Coronal Imager (Hi-C)\citep{Kobayashi2014}.

The ESIS detectors are CCD230-42 astro-process CCDs from E2V.
For each camera, the CCD is operated in a split frame transfer mode with each of the four ports read out by a 16-bit A/D 
converter.
The central $2048 \times 1024$ pixels of the $2k\times2k$ device are used for imaging, while the outer two regions are 
used for storage.
Two overscan columns on either side of the imaging area and eight extra rows in each storage region will monitor read 
noise and dark current.
When the camera receives the trigger signal, it transfers the image from the imaging region to the storage regions and 
starts image readout.
The digitized data are sent to the Data Acquisition and Control System (DACS) through a SpaceWire interface immediately, 
one line at a time.
The frame transfer takes $<$\SI{60}{\milli\second}, and readout takes \SI{1.1}{\second}.
The cadence is adjustable from 2-\SI{600}{\second} in increments of \SI{100}{\milli\second}, to satisfy the requirement 
listed in Table~\ref{table:scireq}.
Because the imaging region is continuously illuminated, the action of frame transfer (transferring the image from the 
imaging region to the storage regions) also starts the next exposure without delay.
Thus the exposure time is controlled by the time period between triggers.
Camera 1 (Fig.~\ref{F-cameras}) generates the sync trigger, which is fed back into Camera 1's trigger input and provides 
independently buffered triggers to the remaining three cameras.
The trigger signals are synchronized to better than $\pm$\SI{1}{\milli\second}.
Shutterless operation allows ESIS to observe with a \SI{100}{\percent} duty cycle.
The cadence is limited only by the 1.1\,s readout time. 

MSFC custom designed the camera board, enclosure, and mounting structure for ESIS to fit the unique packaging 
requirements of this experiment (Fig~\ref{F-cameras}).
The front part of the camera is a metal block which equalizes the temperature across the CCD while fastening it in 
place.
The carriers of all cameras are connected to a central two-piece copper (\SI{3}{\kilo\gram}) and aluminum 
(\SI{1}{\kilo\gram}) thermal reservoir (cold block) by flexible copper cold straps.
The flexible cold straps allow individual cameras to be translated parallel to the optical axis (by means of shims) up 
to $\sim$\SI{13}{\milli\meter} to adjust focus in each channel prior to launch.
The centrally located cold block will be cooled by LN2 flow from outside the payload until just before launch.
The LN2 flow will be controlled automatically by a Ground Support Equipment (GSE) computer so that all cameras are 
maintained above survival temperature but below the target temperature of \SI{-55}{\celsius} to insure a negligible dark 
current level.

The gain, read noise, and dark current of the four cameras were measured at MSFC using an ${}^{55}$Fe radioactive 
source.
Cameras are labeled 1, 2, 3, and 4 with associated serial numbers SN6, SN7, SN9, and SN10 respectively in 
Fig.~\ref{F-cameras}.  Gain ranges from 2.5-\SI{2.6}{e^- \per DN} in each quadrant of all four cameras.
Table~\ref{T-cameras} lists gain, read noise, and dark current by quadrant for each camera.  

The Quantum Efficiency (QE) of the ESIS CCDs will not be measured before flight.
Similar astro-process CCDs with no AR coating are used in the Solar X-ray Imager (SXI) aboard the Geosynchronous 
Orbiting Environmental Satellites (GOES) N and O.
A QE range of 43\% at 583\AA\ to 33\% at 630\AA\ is expected for the ESIS CCDs, based on QE measurements by 
\citet{Stern04} for GOES SXI instruments.

\begin{table}[!htb]
\caption{ESIS Camera properties.}
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
The ESIS DACS are based on the designs used for both CLASP~\citep{Kano12,Kobayashi12} and Hi-C~\citep{Kobayashi2014}.
The electronics are a combination of Military Off-The-Shelf (MOTS) hardware and custom designed components.
The DACS is a 6-slot, 3U, open VPX PCIe architecture conduction cooled system using an AiTech C873 single board
computer.
The data system also include a MOTS PCIe switch card, MSFC parallel interface card, and two MOTS Spacewire cards.
A slot for an additional Spacewire card is included to accommodate two more cameras for the next ESIS flight.
The C873 has a \SI{2.4}{\giga\hertz} Intel i7 processor with \SI{16}{\giga b} of memory.
The operating temperature range for the data system is -40 to +85 C.
The operating system for the flight data system is Linux Fedora 23.

The DACS is responsible for several functions;
it controls the ESIS experiment, responds to timers and uplinks, acquires and stores image data from the cameras, 
downlinks a subset of images through telemetry, and provides experiment health and status.
The DACS is housed with the rest of the avionics (power supply, analog signal conditioning system) in a 
0.56-\SI{0.43}{\meter} transition section outside of the experiment section.
This relaxes the thermal and cleanliness constraints placed on the avionics.
Custom DC/DC converters are used for secondary voltages required by other electronic components.
The use of custom designed converters allowed additional ripple filtering for low noise."""
            ))

        with doc.create(pylatex.Subsection('Pointing System')):
            doc.append(pylatex.NoEscape(
                r"""
The imaging target will be selected prior to launch, the morning of the day of flight.
During flight, pointing will be maintained by the Solar Pointing Attitude Rocket Control System (SPARCS) 
\citep{Lockheed69}.
Images from Camera 1 will be downlinked and displayed in real time on the SPARCS control system console at intervals of 
$\sim$\SI{16}{\second} to verify pointing is maintained during flight."""
            ))

        with doc.create(pylatex.Subsection('Mechanical')):
            doc.append(pylatex.NoEscape(
                r"""
ESIS and MOSES are mounted on opposite sides of a composite optical table structure originally developed for the Solar 
Plasma Diagnostics Experiment rocket mission (SPDE,~\citet{Bruner95lock}).
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
ESIS will be launched aboard a sub-orbital Terrier Black Brant sounding rocket from White Sands Missile Range.
The experiment is currently scheduled for launch in August, 2019.
Trajectory will follow a roughly parabolic path, with $>$\SI{270}{\second} solar observing time above 
\SI{160}{\kilo\meter}.
ESIS will begin continuously taking exposures at a fixed cadence immediately after launch, terminating just before the 
payload impacts the upper atmosphere.
Exposure length will be determined by the target selected for launch day.
Exposures taken while the payload shutter door is closed ($<$ \SI{160}{\kilo\meter}) will be used for dark calibration.
Data will be stored on board and downloaded after recovery, however a limited amount of data will be transmitted to the 
ground station via high speed telemetry as a safeguard against payload loss or destruction.
A parachute will slow the descent of the payload after it enters the atmosphere, and recovery will be accomplished by 
helicopter after the payload is located on the ground."""
        ))

        with doc.create(pylatex.Subsection('ESIS Mission Update')):
            doc.append(pylatex.NoEscape(
                r"""
Since the time of writing ESIS launched and was recovered successfully from White Sands Missile Range on 
September 30, 2019.
Unfortunately, due to failure of the mechanical shutter, no MOSES data was obtained during this flight.
A  paper is forthcoming that will document the ESIS instrument in its as-flown configuration~\citep{Courrier_inprep}.
A companion paper will describe ESIS first results~\citep{Parker_inprep}.
Two significant changes, one to the ESIS instrument and one to our alignment procedures, were made prior to launch and 
are summarized below.

The transfer from visible to EUV grating alignment was completed by an alternative means.
The apparatus described by~\citet{Johnson18} was not able to maintain sufficient repeatability during test runs on 
diffraction grating surfaces.
To maintain the launch schedule, a phase shifting interferometer was used to transfer the alignment of the visible 
gratings to the EUV flight gratings. 

A trade study was conducted, and it was decided to remove the primary aperture stop. The advantage was an increase in 
sensitivity.
The disadvantage was to sacrifice the unvignetted design described in Section \ref{subsec:AperturesandBaffles}.
The effective aperture is increased by a factor of 1.7 to 2.7 as a function of FOV in the radial dimension.
The corresponding signal gradient is oriented along the dispersion direction of each channel;
vignetting increases (and signal decreases) when moving towards blue wavelengths 
(\ie\,moving to the left in Figure~\ref{fig:projections}).
This gradient is due almost entirely to vignetting by the central obscuration, and is linear across the entire FOV.
The principal challenge is that the images cannot be corrected directly;
rather, since the gradient is repeated for each of the overlapping spectral line images, the vignetting can only be 
accounted for by forward modeling.
Since forward modeling is required for all of the inversion procedures under consideration for ESIS data analysis, the 
vignetting was deemed low risk to the mission science."""
            ))

    with doc.create(pylatex.Section('Conclusions and Outlook')):
        doc.append(pylatex.NoEscape(
            r"""
ESIS is a next generation slitless spectrograph, designed to obtain co-temporal spectral and spatial images of the solar 
transition region and corona.
In this report, we present details of the scientific objectives, instrument, image and spectral resolution, data 
acquisition, and flight profile.

ESIS follows on the proven MOSES design, incorporating several design changes to improve the utility of the instrument.
The symmetrical arrangement of CCDs and diffraction gratings results in a compact instrument while increasing the number 
of dispersed images and dispersion planes.
This aids the inversion process, while also allowing access to higher order spectral line profile moments.
Individual gratings improve resolution by controlling aberration in each channel.
The addition of a field stop eliminates spectral contamination and provides an easily recognizable edge for data 
inversion.
The ESIS design also demonstrates that all this can be accomplished in a volume small enough to serve as a prototype for 
a future orbital instrument.

For the first flight, four of the six available ESIS channels will be populated with optics optimized around the 
O\,\textsc{v} emission line.
The large (\SI{11.3}{\arcminute}), high resolution FOV (\SI{1.52}{\arcsecond}, \SI{74}{\milli\angstrom}) can 
simultaneously observe the evolution of small scale EUV flows and large scale MHD waves in high temporal cadence. 
ESIS also enables the study of transport of mass and energy in the transition region and corona during the $\sim 5$ 
minute data collection portion of rocket flight.

ESIS was recovered after a successful first launch on September 30, 2019, with analysis of collected data currently 
in-process.
Subsequent flights will be proposed and the instrument refined with an eye toward orbital opportunities.
Suborbital flights will allow us to expand the instrument to its full complement of six channels and refine our data 
analysis methods, but do not provide access to major flares and eruptive events that drive space weather.
The long term prospect is that an ESIS-like instrument on an orbital platform could provide high cadence maps of 
spectral line profiles in solar flares, allowing unique and comprehensive observations of the dynamics in solar eruptive 
events, flare ribbons, and the flare reconnection region."""
        ))

    doc.append(pylatex.Command('bibliography', arguments='sources'))

    return doc
