import astropy.units as u
import pylatex
import kgpy.latex
import esis.flight
from .. import optics

__all__ = [
    'table_old',
]


def table_old() -> pylatex.Table:
    result = pylatex.Table()
    result._star_latex_name = True
    with result.create(pylatex.Center()) as centering:
        with centering.create(pylatex.Tabular('llrr')) as tabular:
            tabular.escape = False
            tabular.add_row([r'Element', r'Parameter', r'Requirement', r'Measured'])
            tabular.add_hline()
            tabular.add_row([r'Primary', r'RMS slope error ($\mu$rad)', r'$<1.0$', r''])
            tabular.add_row([r'', r'Integration length (mm)', r'4.0', r''])
            tabular.add_row([r'', r'Sample length (mm)', r'2.0', r''])
            tabular.add_hline()
            tabular.add_row([r'Primary', r'RMS roughness (nm)', r'$<2.5$', r''])
            tabular.add_row([r'', r'Periods (mm)', r'0.1-6', r''])
            tabular.add_hline()
            tabular.add_row([r'Grating', r'RMS slope error ($\mu$rad)', r'$<3.0$', r''])
            tabular.add_row([r'', r'Integration length (mm)', r'2 \roy{why fewer sigfigs?}', r''])
            tabular.add_row([r'', r'Sample length (mm)', r'1', r''])
            tabular.add_hline()
            tabular.add_row([r'Grating', r'RMS roughness (nm)', r'$<2.3$', r''])
            tabular.add_row([r'', r'Periods (mm)', r'0.02-2', r''])
            tabular.add_hline()
    result.add_caption(pylatex.NoEscape(
        r"""
Figure and surface roughness requirements compared to metrology for the \ESIS\ optics.
Slope error (both the numerical estimates and the measurements) is worked out with integration length and sample length 
defined per ISO 10110."""
    ))
    result.append(kgpy.latex.Label('table:error'))
    return result


def table() -> pylatex.Table:
    optics_single = optics.as_designed_single_channel()
    optics_all = esis.flight.optics.as_measured()
    primary = optics_single.primary
    grating = optics_single.grating
    unit_length_integration = u.mm
    unit_length_sample = u.mm
    unit_slope_error = u.urad
    unit_ripple_period = u.mm
    unit_ripple = u.nm
    unit_microroughness_period = u.um
    unit_microroughness = u.nm
    result = pylatex.Table()
    result._star_latex_name = True
    with result.create(pylatex.Center()) as centering:
        with centering.create(pylatex.Tabular('llrr')) as tabular:
            tabular.escape = False
            tabular.add_row([
                r'Element',
                r'Parameter',
                r'Requirement',
                r'Measured',
            ])
            tabular.add_hline()
            tabular.add_row([
                r'Primary',
                f'RMS slope error ({unit_slope_error:latex_inline})',
                f'{optics_single.primary.slope_error.value.to(unit_slope_error).value:0.1f}',
                f'{optics_all.primary.slope_error.value.to(unit_slope_error).value:0.1f}',
            ])
            tabular.add_row([
                r'',
                f'\\quad Integration length = {primary.slope_error.length_integration.to(unit_length_integration).value:0.1f}\,{unit_length_integration:latex_inline}',
                r'',
                r'',
            ])
            tabular.add_row([
                r'',
                f'\\quad Sample length = {primary.slope_error.length_sample.to(unit_length_sample).value:0.1f}\,{unit_length_sample:latex_inline}',
                r'',
                r'',
            ])
            tabular.add_row([
                r'',
                f'RMS roughness ({unit_ripple:latex_inline})',
                f'{optics_single.primary.ripple.value.to(unit_ripple).value:0.1f}',
                f'{optics_all.primary.ripple.value.to(unit_ripple).value:0.1f}',
            ])
            tabular.add_row([
                r'',
                f'\quad Periods = ${primary.ripple.periods_min.to(unit_ripple_period).value:0.2f}-{primary.ripple.periods_max.to(unit_ripple_period).value:0.1f}$\\,{unit_ripple_period:latex_inline}',
                r'',
                r'',
            ])
            tabular.add_row([
                r'',
                f'RMS microroughness ({unit_microroughness:latex_inline})',
                f'{optics_single.primary.microroughness.value.to(unit_ripple).value:0.1f}',
                f'{optics_all.primary.microroughness.value.to(unit_ripple).value:0.1f}',
            ])
            tabular.add_row([
                r'',
                f'\quad Periods = ${primary.microroughness.periods_min.to(unit_microroughness_period).value:0.2f}-{primary.microroughness.periods_max.to(unit_microroughness_period).value:0.1f}$\\,{unit_microroughness_period:latex_inline}',
                r'',
                r'',
            ])
            tabular.add_hline()
            tabular.add_row([
                r'Grating',
                f'RMS slope error ({unit_slope_error:latex_inline})',
                f'{optics_single.grating.slope_error.value.to(unit_slope_error).value:0.1f}',
                f'{optics_all.grating.slope_error.value.to(unit_slope_error).value.mean():0.1f}',
            ])
            tabular.add_row([
                r'',
                f'\\quad Integration length = {grating.slope_error.length_integration.to(unit_length_integration).value:0.1f}\,{unit_length_integration:latex_inline}',
                r'',
                r'',
            ])
            tabular.add_row([
                r'',
                f'\\quad Sample length = {grating.slope_error.length_sample.to(unit_length_sample).value:0.1f}\,{unit_length_sample:latex_inline}',
                r'',
                r'',
            ])
            tabular.add_row([
                r'',
                f'RMS roughness ({unit_ripple:latex_inline})',
                f'{optics_single.grating.ripple.value.to(unit_ripple).value:0.1f}',
                f'{optics_all.grating.ripple.value.to(unit_ripple).value.mean():0.1f}',
            ])
            tabular.add_row([
                r'',
                f'\quad Periods = ${grating.ripple.periods_min.to(unit_ripple_period).value:0.2f}-{grating.ripple.periods_max.to(unit_ripple_period).value:0.1f}$\\,{unit_ripple_period:latex_inline}',
                r'',
                r'',
            ])
            tabular.add_row([
                r'',
                f'RMS microroughness ({unit_microroughness:latex_inline})',
                f'{optics_single.grating.microroughness.value.to(unit_ripple).value:0.1f}',
                f'{optics_all.grating.microroughness.value.to(unit_ripple).value.mean():0.1f}',
            ])
            tabular.add_row([
                r'',
                f'\quad Periods = ${grating.microroughness.periods_min.to(unit_microroughness_period).value:0.2f}-{grating.microroughness.periods_max.to(unit_microroughness_period).value:0.1f}$\\,{unit_microroughness_period:latex_inline}',
                r'',
                r'',
            ])
            tabular.add_hline()

    result.add_caption(pylatex.NoEscape(
        r"""
Figure and surface roughness requirements compared to metrology for the \ESIS\ optics.
Slope error (both the numerical estimates and the measurements) is worked out with integration length and sample length 
defined per ISO 10110."""
    ))
    result.append(kgpy.latex.Label('table:error'))
    return result
