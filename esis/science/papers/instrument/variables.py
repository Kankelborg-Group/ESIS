import numpy as np
import astropy.units as u
import pylatex
import num2words
import kgpy.latex
import kgpy.chianti
import esis.optics
from . import optics
from . import figures

__all__ = [
    'append_to_document',
]


def append_to_document(doc: kgpy.latex.Document):
    wavl_digits = 2

    doc.set_variable(
        name='ie',
        value=pylatex.NoEscape(r'\textit{i.e.}')
    )

    doc.set_variable(
        name='eg',
        value=pylatex.NoEscape(r'\textit{e.g.}')
    )

    doc.set_variable('spiejatis', pylatex.NoEscape(r'J~.Atmos.~Tel. \& Img.~Sys.'))

    requirements = esis.optics.design.requirements()

    optics_single = optics.as_designed_single_channel()

    optics_single_m2 = esis.optics.design.final(all_channels=False)
    optics_single_m2.grating.diffraction_order = 2

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
        value=pylatex.NoEscape(r'\OVion~\OVwavelength')
    )

    index_he1 = np.nonzero(optics_single.bunch.ion == 'he_1')[0][0]
    wavelength_he1 = wavelength[index_he1]
    doc.set_variable_quantity(
        name='HeIwavelength',
        value=wavelength_he1,
        digits_after_decimal=wavl_digits,
    )
    doc.set_variable(
        name='HeIion',
        value=pylatex.NoEscape(ion[index_he1])
    )
    doc.set_variable(
        name='HeI',
        value=pylatex.NoEscape(r'\HeIion~\HeIwavelength')
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
        value=pylatex.NoEscape(r'\MgXion~\MgXwavelength')
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
        value=pylatex.NoEscape(r'\MgXdimIon~\MgXdimWavelength')
    )

    wavelength_m2 = optics_single_m2.bunch.wavelength
    ion_m2 = kgpy.chianti.to_spectroscopic(optics_single_m2.bunch.ion)
    index_he2 = np.nonzero(optics_single_m2.bunch.ion == 'he_2')[0][0]
    wavelength_he2 = wavelength_m2[index_he2]
    doc.set_variable_quantity(
        name='HeIIwavelength',
        value=wavelength_he2,
        digits_after_decimal=wavl_digits,
    )
    doc.set_variable(
        name='HeIIion',
        value=pylatex.NoEscape(ion_m2[index_he2])
    )
    doc.set_variable(
        name='HeII',
        value=pylatex.NoEscape(r'\HeIIion~\HeIIwavelength')
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

    channel_names_str = ''
    for i, cname in enumerate(optics_all.channel_name):
        if i + 1 == optics_all.channel_name.shape[~0]:
            channel_names_str += f'and {cname}'
        else:
            channel_names_str += f'{cname}, '
    doc.set_variable(
        name='channelNames',
        value=channel_names_str,
    )

    doc.set_variable(
        name='fNumber',
        value=pylatex.NoEscape(f'$f$/{int(optics_single.f_number_effective)}'),
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
    dz = optics_single.detector.translation.z - optics_single.grating.translation.z
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
        name='plateScaleMean',
        value=optics_single.plate_scale.quantity.mean(),
        digits_after_decimal=2,
    )

    doc.set_variable_quantity(
        name='spatialResolution',
        value=optics_single.resolution_spatial.quantity,
        digits_after_decimal=2,
    )

    doc.set_variable_quantity(
        name='spatialResolutionMax',
        value=optics_single.resolution_spatial.quantity.max(),
        digits_after_decimal=2,
    )

    doc.set_variable_quantity(
        name='dispersion',
        value=optics_single.dispersion.to(kgpy.units.mAA / u.pix),
        digits_after_decimal=1,
    )

    doc.set_variable_quantity(
        name='spectralResolution',
        value=(2 * u.pix) * optics_single.dispersion.to(kgpy.units.mAA / u.pix),
        digits_after_decimal=0,
    )

    km_per_s_per_pix = u.def_unit(
        'km_per_s_per_pix',
        represents=u.km / u.s / u.pix,
        format=dict(
            latex=r'\mathrm{km\,s^{-1}\,pix^{-1}}'
        )
    )
    doc.set_variable_quantity(
        name='dispersionDoppler',
        value=optics_single.dispersion_doppler.to(km_per_s_per_pix),
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

    doc.set_variable_quantity(
        name='primaryFocalLengthMeasured',
        value=optics_all.primary.focal_length,
        digits_after_decimal=1,
    )

    doc.set_variable_quantity(
        name='primaryMtfDegradationFactor',
        value=optics_single.primary.mtf_degradation_factor,
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
    efficiency_primary = optics_single.primary.material.transmissivity(rays_o5).to(u.percent)
    doc.set_variable_quantity(
        name='primaryEfficiency',
        value=efficiency_primary,
        digits_after_decimal=0,
    )

    rays_he2 = kgpy.optics.rays.Rays(wavelength=wavelength_he2)
    efficiency_primary_he2 = optics_single.primary.material.transmissivity(rays_he2).to(u.percent)
    rejection_primary_he2 = efficiency_primary_he2 / efficiency_primary

    efficiency_measured_primary = optics_all.primary.material.transmissivity(rays_o5).to(u.percent)
    doc.set_variable_quantity(
        name='primaryEfficiencyMeasured',
        value=efficiency_measured_primary,
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
        digits_after_decimal=5,
    )

    doc.set_variable_quantity(
        name='gratingRulingSpacingMeasured',
        value=optics_all.grating.surface.rulings.ruling_spacing.to(u.um),
        digits_after_decimal=5,
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

    grating_radius_min = optics_all.grating.tangential_radius.min()
    grating_radius_max = optics_all.grating.tangential_radius.max()
    grating_radius_mean = (grating_radius_max + grating_radius_min) / 2
    grating_radius_range = (grating_radius_max - grating_radius_min) / 2
    doc.set_variable(
        name='gratingRadiusMeasured',
        value=pylatex.NoEscape(
            f'${grating_radius_mean.value:0.3f}\\pm{grating_radius_range.value:0.3f}$\\,'
            f'{grating_radius_mean.unit:latex_inline}'
        )
    )

    doc.set_variable_quantity(
        name='gratingMtfDegradationFactor',
        value=optics_single.grating.mtf_degradation_factor,
        digits_after_decimal=1,
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

    efficency_grating = optics_all.grating.material.transmissivity(rays_o5).to(u.percent)
    doc.set_variable_quantity(
        name='gratingEfficiency',
        value=efficency_grating,
        digits_after_decimal=0,
    )

    efficiency_grating_witness = optics_all.grating.witness.transmissivity(rays_o5).to(u.percent)
    doc.set_variable_quantity(
        name='gratingWitnessEfficiency',
        value=efficiency_grating_witness,
        digits_after_decimal=0,
    )

    unmeasured = esis.optics.grating.efficiency.witness.manufacturing_number_unmeasured
    doc.set_variable(
        name='gratingWitnessMissingChannel',
        value=optics_all.channel_name[np.nonzero(optics_all.grating.manufacturing_number == unmeasured)][0]
    )

    efficiency_grating_witness_he2 = optics_all.grating.witness.transmissivity(rays_he2).to(u.percent)
    rejection_grating_he2 = (efficiency_grating_witness_he2 / efficiency_grating_witness)
    doc.set_variable_quantity(
        name='gratingHeIIRejectionRatio',
        value=rejection_grating_he2.to(u.percent),
        digits_after_decimal=0,
    )

    doc.set_variable_quantity(
        name='gratingGrooveEfficiency',
        value=(efficency_grating / efficiency_grating_witness).to(u.percent),
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
        name='filterClocking',
        value=optics_single.filter.clocking,
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

    doc.set_variable_quantity(
        name='filterMeshPitch',
        value=optics_single.filter.mesh_pitch,
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

    efficiency_filter = optics_single.filter.surface.material.transmissivity(rays_o5).to(u.percent)
    doc.set_variable_quantity(
        name='filterEfficiency',
        value=efficiency_filter,
        digits_after_decimal=0,
    )

    efficiency_filter_he2 = optics_single.filter.surface.material.transmissivity(rays_he2).to(u.percent)
    rejection_filter_he2 = efficiency_filter_he2 / efficiency_filter

    doc.set_variable_quantity(
        name='filterToDetectorDistance',
        value=optics_single.filter.translation.z - optics_single.detector.translation.z,
        digits_after_decimal=0,
    )

    doc.set_variable(
        name='detectorName',
        value=str(optics_single.detector.name),
    )

    doc.set_variable(
        name='detectorManufacturer',
        value=optics_single.detector.manufacturer,
    )

    detector_serial_numbers = ''
    for i, cname in enumerate(optics_all.detector.serial_number):
        if i + 1 == optics_all.detector.serial_number.shape[~0]:
            detector_serial_numbers += f'and {cname}'
        else:
            detector_serial_numbers += f'{cname}, '
    doc.set_variable(
        name='detectorSerialNumbers',
        value=detector_serial_numbers,
    )

    doc.set_variable_quantity(
        name='detectorFocusAdjustmentRange',
        value=optics_single.detector.range_focus_adjustment,
        digits_after_decimal=0,
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
        name='detectorTemperatureTarget',
        value=optics_single.detector.temperature,
        digits_after_decimal=0,
    )

    doc.set_variable(
        name='detectorGainRange',
        value=pylatex.NoEscape(
            f'{optics_all.detector.gain.value.min():0.1f}\\nobreakdash-{optics_all.detector.gain.value.max():0.1f}\\,'
            f'{optics_all.detector.gain.unit:latex_inline}'
        ),
    )

    doc.set_variable(
        name='detectorNumOverscanColumns',
        value=str(optics_single.detector.npix_overscan),
    )

    doc.set_variable(
        name='detectorNumOverscanColumnWords',
        value=num2words.num2words(optics_single.detector.npix_overscan),
    )

    doc.set_variable(
        name='DetectorNumOverscanColumnWords',
        value=num2words.num2words(optics_single.detector.npix_overscan).title(),
    )

    efficiency_detector = optics_single.detector.surface.material.transmissivity(rays_o5).to(u.percent)
    doc.set_variable_quantity(
        name='detectorQuantumEfficiency',
        value=efficiency_detector,
        digits_after_decimal=0
    )

    efficiency_detector_he2 = optics_single.detector.surface.material.transmissivity(rays_he2).to(u.percent)
    rejection_detector_he2 = efficiency_detector_he2 / efficiency_detector

    rays_he1 = kgpy.optics.rays.Rays(wavelength=wavelength_he1)
    doc.set_variable_quantity(
        name='detectorQuantumEfficiencyHeI',
        value=optics_single.detector.surface.material.transmissivity(rays_he1).to(u.percent),
        digits_after_decimal=0
    )

    doc.set_variable_quantity(
        name='detectorFrameTransferTime',
        value=optics_single.detector.time_frame_transfer,
        digits_after_decimal=0
    )

    doc.set_variable_quantity(
        name='detectorReadoutTime',
        value=optics_single.detector.time_readout,
        digits_after_decimal=1,
    )

    doc.set_variable_quantity(
        name='detectorExposureLength',
        value=optics_single.detector.exposure_length,
        digits_after_decimal=0,
    )

    doc.set_variable_quantity(
        name='detectorMinExposureLength',
        value=optics_single.detector.exposure_length_min,
        digits_after_decimal=1,
    )

    doc.set_variable(
        name='detectorMinExposureLengthValue',
        value=f'{optics_single.detector.exposure_length_min.value:0.0f}',
    )

    doc.set_variable_quantity(
        name='detectorMaxExposureLength',
        value=optics_single.detector.exposure_length_max,
        digits_after_decimal=0,
    )

    doc.set_variable(
        name='detectorExposureLengthRange',
        value=pylatex.NoEscape(r'\detectorMinExposureLengthValue-\detectorMaxExposureLength'),
    )

    doc.set_variable_quantity(
        name='detectorExposureLengthIncrement',
        value=optics_single.detector.exposure_length_increment,
        digits_after_decimal=0,
    )

    doc.set_variable(
        name='detectorTriggerIndex',
        value=str(optics_all.channel_name[optics_all.detector.index_trigger]),
    )

    doc.set_variable_quantity(
        name='detectorSynchronizationError',
        value=optics_single.detector.error_synchronization,
        digits_after_decimal=0,
    )

    doc.set_variable(
        name='detectorAnalogToDigitalBits',
        value=str(optics_single.detector.bits_analog_to_digital)
    )

    doc.set_variable_quantity(
        name='totalEfficiency',
        value=(efficiency_measured_primary * efficency_grating * efficiency_filter * efficiency_detector).to(u.percent),
        digits_after_decimal=2,
    )

    rejection_he2 = rejection_primary_he2 * rejection_grating_he2 * rejection_filter_he2 * rejection_detector_he2
    doc.set_variable_quantity(
        name='totalHeIIRejection',
        value=10 * np.log10(rejection_he2) * u.dB,
        digits_after_decimal=0,
    )

    doc.set_variable(
        name='defaultPupilSamples',
        value=optics.default_kwargs['pupil_samples'],
    )

    doc.set_variable(
        name='defaultFieldSamples',
        value=optics.default_kwargs['field_samples']
    )

    doc.set_variable(
        name='defaultNumEmissionLines',
        value=optics.num_emission_lines_default,
    )

    doc.set_variable(
        name='defaultNumEmissionLinesWords',
        value=num2words.num2words(optics.num_emission_lines_default),
    )

    doc.set_variable(
        name='psfPupilSamples',
        value=figures.psf.pupil_samples,
    )

    doc.set_variable(
        name='psfFieldSamples',
        value=figures.psf.field_samples,
    )

    doc.set_variable_quantity(
        name='skinDiameter',
        value=optics_single.skin_diameter.to(u.m),
        digits_after_decimal=1,
    )

    doc.set_variable(
        name='C',
        value=pylatex.NoEscape(r'\mathbf{C}')
    )
