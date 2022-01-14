import pylatex
import kgpy.latex

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
