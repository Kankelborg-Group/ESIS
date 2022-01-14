import astropy.units as u
import pylatex
import kgpy.format
from .. import optics

__all__ = [
    'table',
]


def table() -> pylatex.Table:
    optics_single = optics.as_designed_single_channel()
    model_distortion = optics_single.rays_output.distortion.model()
    model_distortion_relative = optics_single.rays_output_relative.distortion.model()

    def fmt_coeff(coeff: u.Quantity):
        return kgpy.format.quantity(
            a=coeff.value * u.dimensionless_unscaled,
            scientific_notation=True,
            digits_after_decimal=2,
        )

    result = pylatex.Table()
    with result.create(pylatex.Center()) as centering:
        with centering.create(pylatex.Tabular('ll|rr')) as tabular:
            tabular.escape = False
            tabular.append('\multicolumn{2}{l}{Coefficient} & $x\'$ & $y\'$\\\\')
            tabular.add_hline()
            for c, name in enumerate(model_distortion.x.coefficient_names):
                tabular.add_row([
                    f'{name}',
                    f'({model_distortion.x.coefficients[c].unit:latex_inline})',
                    fmt_coeff(model_distortion_relative.x.coefficients[c].squeeze()),
                    fmt_coeff(model_distortion_relative.y.coefficients[c].squeeze()),
                ])
    return result
