import typing as typ
import pathlib
import numpy as np
import matplotlib.lines
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import pylatex
import kgpy.latex
import kgpy.format
import esis.optics
from .. import optics as optics_factories
from . import formatting
from . import caching

__all__ = [
    'annotate_wavelength',
    'figure_mpl',
    'pdf',
    'figure',
]


def annotate_wavelength(
        ax: matplotlib.axes.Axes,
        label_orders: bool = True
) -> typ.List[matplotlib.lines.Line2D]:
    optics = esis.optics.design.final(**optics_factories.default_kwargs)
    optics.num_emission_lines = 2
    optics2 = optics.copy()
    optics2.grating.diffraction_order = 2
    optics2.num_emission_lines = 2

    optics.plot_wavelength_range(ax=ax)
    optics2.plot_wavelength_range(ax=ax)
    if label_orders:
        ax.text(
            x=(optics.wavelength_min + optics.wavelength_max) / 2,
            y=1.01,
            s='$m=1$',
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='bottom',
        )
        ax.text(
            x=(optics2.wavelength_min + optics2.wavelength_max) / 2,
            y=1.01,
            s='$m=2$',
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='bottom',
        )
    lines = optics.bunch.plot_wavelength(
        ax=ax,
        num_emission_lines=optics.num_emission_lines,
        digits_after_decimal=formatting.digits_after_decimal,
        colors=['orangered', 'blue'],
    )
    lines += optics2.bunch.plot_wavelength(
        ax=ax,
        num_emission_lines=optics2.num_emission_lines,
        digits_after_decimal=formatting.digits_after_decimal,
        colors=['gray', 'black'],
    )
    return lines


def figure_mpl() -> matplotlib.figure.Figure:
    fig, axs = plt.subplots(
        nrows=2,
        sharex=True,
        figsize=(formatting.column_width, 5),
        constrained_layout=True,
    )
    eff_unit = u.percent
    wavl_unit = u.Angstrom
    witness = esis.optics.grating.efficiency.witness
    with astropy.visualization.quantity_support():
        optics_design = esis.optics.design.final()
        optics = esis.flight.optics.as_measured()
        for func in [witness.vs_wavelength_g24, witness.vs_wavelength_g17, witness.vs_wavelength_g19, ]:
            serial, angle_input, wavelength, efficiency = func()
            chan_i = np.nonzero(optics.grating.manufacturing_number == serial)
            axs[0].plot(
                wavelength.to(wavl_unit),
                efficiency.to(eff_unit),
                label=f'Channel {optics.channel_name[chan_i].squeeze()}',
            )
        axs[0].legend()

        axs[0].set_xlabel(None)
        axs[0].set_ylabel(f'efficiency ({eff_unit:latex})')
        annotate_wavelength(ax=axs[0])

        wavelength = esis.optics.grating.efficiency.witness.vs_wavelength_g17()[2].to(wavl_unit)
        rays = kgpy.optics.rays.Rays(
            wavelength=wavelength,
        )

        axs[1].plot(
            wavelength,
            optics.primary.material.transmissivity(rays).to(eff_unit),
            label=f'primary',
            color='tab:red'
        )

        axs[1].plot(
            wavelength,
            optics_design.primary.material.transmissivity(rays).to(eff_unit),
            label=f'primary model',
            color='tab:red',
            alpha=0.5,
        )

        axs[1].plot(
            wavelength,
            optics.grating.material.transmissivity(rays).to(eff_unit),
            label=f'grating',
            color='tab:purple'
        )

        axs[1].plot(
            wavelength,
            optics.filter.surface.material.transmissivity(rays).to(u.percent),
            label=r'filter',
            color='tab:cyan',
        )

        axs[1].add_artist(axs[1].legend())
        lines = annotate_wavelength(ax=axs[1], label_orders=False)
        axs[1].set_xlabel(f'wavelength ({wavl_unit:latex})')
        axs[1].set_ylabel(f'efficiency ({eff_unit:latex})')
        axs[1].legend(handles=lines, bbox_to_anchor=(0.5, -0.25), loc='upper center', ncol=2)

    return fig


def pdf() -> pathlib.Path:
    return caching.cache_pdf(figure_mpl)


def figure() -> pylatex.Figure:
    result = pylatex.Figure()
    result.add_image(str(pdf()), width=None)
    result.add_caption(pylatex.NoEscape(
        r"""
(Top) Measured reflectance for several multilayer coated witness samples 
\roy{at an incidence angle of \gratingWitnessMeasurementIncidenceAngle\ on \testGratingDate.
The white regions indicate wavelengths that intercept the detector and the gray regions indicate wavelengths that
miss the detector.
Note the suppression of second order relative to the first order and the consistency of the coatings between each 
channel.
The Channel \gratingWitnessMissingChannel\ grating measurement is missing due to issues in the measurement apparatus.
(Bottom) Comparison of the efficiency of the three main \ESIS\ optical components: primary mirror, grating and filter.
The primary mirror efficiency is based on measurements of a \Si\ witness sample taken on \primaryMeasurementDate\ at an 
angle of incidence of \primaryWitnessMeasurementIncidenceAngle. 
The grating efficiency is from a measurement of the Channel \testGratingChannelIndex\ grating taken on \testGratingDate\
at an angle of incidence of \gratingMeasurementIncidenceAngle.
The filter efficiency is a theoretical model that includes the filter mesh, \filterThickness\ of \filterMaterial\ and
\filterOxideThickness\ of \filterMaterial\ oxide.}"""
    ))
    result.append(kgpy.latex.Label('fig:componentEfficiencyVsWavelength'))
    return result
