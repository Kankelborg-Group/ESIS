import pathlib
import matplotlib.figure
import matplotlib.pyplot as plt
import astropy.units as u
import pylatex
import kgpy.latex
import kgpy.format
import esis.optics
from .. import optics as optics_factories
from . import formatting
from . import caching

__all__ = [
    'figure_mpl',
    'pdf',
    'figure',
]


def figure_mpl() -> matplotlib.figure.Figure:
    optics = esis.optics.design.final(**optics_factories.default_kwargs)
    fig, ax = plt.subplots(
        figsize=(formatting.column_width, 2),
        constrained_layout=True,
    )
    eff_unit = u.percent
    output_angle_unit = u.deg
    for func in [esis.optics.grating.efficiency.vs_angle_at_0aoi, esis.optics.grating.efficiency.vs_angle_at_3aoi]:
        input_angle, output_angle, wavelength, eff = func()
        ax.plot(
            output_angle.to(output_angle_unit),
            eff.to(eff_unit),
            label=f'input angle = {kgpy.format.quantity(input_angle, digits_after_decimal=0)}'
        )
    angle_m0 = 0 * u.deg
    angle_m1 = optics.grating.diffraction_angle(optics.wavelength[0]).to(u.deg)
    ax.axvline(angle_m0.value, linestyle='dashed', color='black')
    ax.axvline(angle_m1.value, linestyle='dashed', color='black')
    ax.text(
        x=angle_m0.value,
        y=1.01,
        s='$m=0$',
        transform=ax.get_xaxis_transform(),
        ha='center',
        va='bottom',
    )
    ax.text(
        x=angle_m1.value,
        y=1.01,
        s='$m=1$',
        transform=ax.get_xaxis_transform(),
        ha='center',
        va='bottom',
    )
    ax.set_xlabel(f'output angle ({output_angle_unit:latex})')
    ax.set_ylabel(f'efficiency ({eff_unit:latex})')
    ax.legend()

    return fig


def pdf() -> pathlib.Path:
    return caching.cache_pdf(figure_mpl)


def figure() -> pylatex.Figure:
    result = pylatex.Figure()
    result.add_image(str(pdf()), width=None)
    result.add_caption(pylatex.NoEscape(
        r"""
Measured efficiency \roy{at \gratingTestWavelength} of a single grating \roy{the Channel \testGratingChannelIndex\ grating} as a function of reflection angle on \roy{\testGratingDate}.
Note flat response in first order over instrument \FOV\ and suppression of zero order."""
    ))
    result.append(kgpy.latex.Label('fig:gratingEfficiencyVsAngle'))
    return result
