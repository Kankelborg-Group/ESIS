import pylatex
import kgpy.latex

__all__ = [
    'table'
]


def table() -> pylatex.Table:
    result = pylatex.Table()
    result._star_latex_name = True
    with result.create(pylatex.Center()) as centering:
        with centering.create(pylatex.Tabular('lll')) as tabular:
            tabular.escape = False
            tabular.add_row(['Surface', 'Parameter', 'Calculated value (measured value)'])
            tabular.add_hline()
            tabular.add_row([r'Primary', r'Surface shape', r'Parabolic'])
            tabular.add_row([r'', r'Focal length ', r'\primaryFocalLength\ (\primaryFocalLengthMeasured)'])
            tabular.add_row([r'', r'Aperture shape', r'Octagonal'])
            tabular.add_row([r'', r'Aperture diameter', r'\primaryDiameter'])
            tabular.add_row([r'', r'Coating', r'\primaryCoatingMaterialShort\ single layer, optimized for \OV'])
            tabular.add_row([r'', r'Efficiency (\OV)', r'\primaryEfficiency\ (\primaryEfficiencyMeasured)'])

            tabular.add_hline()
            tabular.add_row([r'Field stop', r'Sky plane diameter', r'\fov'])
            tabular.add_row([r'', r'Aperture shape', r'Octagonal'])
            tabular.add_row([r'', r'Aperture diameter', r'\fieldStopDiameter'])

            tabular.add_hline()
            tabular.add_row([r'Gratings (\numChannels)', r'Surface shape', r'Spherical'])
            tabular.add_row([r'', r'Surface radius', r'\gratingRadius\ (\gratingRadiusMeasured)'])
            tabular.add_row([r'', r'Aperture shape', r'Trapezoidal'])
            tabular.add_row([r'', r'Aperture height', r'\gratingHeight'])
            tabular.add_row([r'', r'Aperture long base', r'\gratingLongWidth'])
            tabular.add_row([r'', r'Aperture short base', r'\gratingShortWidth'])
            tabular.add_row([r'', r'Ruling type', r'Varied line spacing'])
            tabular.add_row([r'', r'Constant ruling spacing coefficient', r'\gratingRulingSpacing\ (\gratingRulingSpacingMeasured)'])
            tabular.add_row([r'', r'Linear ruling spacing coefficient', r'\gratingLinearRulingSpacingCoefficient'])
            tabular.add_row([r'', r'Quadratic ruling spacing coefficient', r'\gratingQuadraticRulingSpacingCoefficient'])
            tabular.add_row([r'', r'Input angle', r'\gratingInputAngle'])
            tabular.add_row([r'', r'Output angle (\OV)', r'\gratingOutputAngle'])
            tabular.add_row([r'', r'Manufacturing process', r'Individual master gratings'])
            tabular.add_row([r'', r'Coating', r'\gratingCoatingMaterialShort\ multilayer, optimized for \OV'])
            tabular.add_row([r'', r'Groove efficiency (\OV)', r'(\gratingGrooveEfficiency)'])
            tabular.add_row([r'', r'Efficiency (\OV)', r'(\gratingEfficiency)'])

            tabular.add_hline()
            tabular.add_row([r'Filters (\numChannels)', r'Aperture shape', r'Circular'])
            tabular.add_row([r'', r'Aperture diameter', r'\filterDiameter'])
            tabular.add_row([r'', r'Material', r'\filterMaterial'])
            tabular.add_row([r'', r'Thickness', r'\filterThickness'])
            tabular.add_row([r'', r'Mesh ratio', r'\filterMeshRatio'])
            tabular.add_row([r'', r'Mesh material', r'\filterMeshMaterialShort'])
            tabular.add_row([r'', r'Efficiency (\OV)', r'\filterEfficiency'])

            tabular.add_hline()
            tabular.add_row([r'Detectors (\numChannels)', r'Manufacturer', r'\detectorManufacturer'])
            tabular.add_row([r'', r'Model', r'\detectorName'])
            tabular.add_row([r'', r'Active area', r'\detectorPixelsX\ $\times$ \detectorPixelsY'])
            tabular.add_row([r'', r'Pixel size', r'\detectorPixelSize'])
            tabular.add_row([r'', r'Quantum efficiency (\OV)', r'\detectorQuantumEfficiency'])
            tabular.add_row([r'', r'Minumum cadence', r'\detectorMinExposureLength'])

            tabular.add_hline()
            tabular.add_row([r'System', r'Magnification', r'\magnification'])
            tabular.add_row([r'', r'$f$-number', r'\fNumber'])
            tabular.add_row([r'', r'Plate scale', r'\plateScale'])
            tabular.add_row([r'', r'Nyquist resolution', r'\spatialResolution'])
            tabular.add_row([r'', r'Dispersion', r'\dispersion\ (\dispersionDoppler)'])
            tabular.add_row([r'', r'Passband', r'\minWavelength\ to \maxWavelength'])
            tabular.add_row([r'', r'Efficiency', r'\totalEfficiency'])
            # tabular.add_row([r'Back focal length', r'\SI{127}{\milli\meter}'])

            tabular.add_hline()

    result.add_caption(pylatex.NoEscape(r"""\ESIS\ design parameters."""))
    result.append(kgpy.latex.Label('table:prescription'))
    return result
