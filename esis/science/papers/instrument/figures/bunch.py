import pathlib
import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import pylatex
import kgpy.grid
import kgpy.latex
import esis
from . import formatting
from . import caching
from .. import optics as optics_factories

__all__ = [
    'figure_mpl',
    'pdf',
]


def figure_mpl() -> matplotlib.figure.Figure:
    with astropy.visualization.quantity_support():
        fig, ax = plt.subplots(figsize=(formatting.text_width, 2), constrained_layout=True)
        optics = esis.optics.design.final()
        optics.bunch.plot(
            ax=ax,
            num_emission_lines=optics.num_emission_lines,
            digits_after_decimal=formatting.digits_after_decimal,
            label_fontsize=6,
        )
        ax_twin = ax.twinx()
        optics_measured = esis.flight.optics.as_measured(**optics_factories.default_kwargs)
        wavelength_min = optics_measured.wavelength_min
        wavelength_max = optics_measured.wavelength_max
        optics_measured.filter.clear_radius = 1000 * u.mm
        optics_measured.detector.num_pixels = (4096, 2048)
        sys = optics_measured.system
        sys.grid_rays.wavelength = kgpy.grid.RegularGrid1D(
            min=wavelength_min,
            max=wavelength_max,
            num_samples=100,
        )
        rays = sys.rays_output_resample_entrance
        area = rays.intensity.copy()
        area[~rays.mask] = np.nan
        area = np.nansum(area, (rays.axis.pupil_x, rays.axis.pupil_y, rays.axis.velocity_los), keepdims=True)
        area[area == 0] = np.nan
        # plt.figure()
        # plt.imshow(area.squeeze()[..., 0])
        # plt.show()
        area = np.nanmean(area, (rays.axis.field_x, rays.axis.field_y)).squeeze()
        subtent = optics_measured.system.rays_input.input_grid.field.step_size
        # area = (area / subtent.x / subtent.y).to(u.cm ** 2)

        wavelength = rays.wavelength.squeeze()
        sorted_indices = np.argsort(wavelength)
        area = area[sorted_indices]
        wavelength = wavelength[sorted_indices]
        ax_twin.plot(wavelength, area, color='red', zorder=0)
        bottom, top = ax_twin.get_ylim()
        margin = 0.05 * (top - bottom)
        ax_twin.set_ylim(bottom=-margin, top=top + 4 * margin)
        ax_twin.set_ylabel(f'mean effective area ({ax_twin.get_ylabel()})')

    return fig


def pdf() -> pathlib.Path:
    return caching.cache_pdf(figure_mpl)


def figure() -> pylatex.Figure:
    result = pylatex.Figure(position='htb!')
    result._star_latex_name = True
    result.add_image(
        filename=str(pdf()),
        width=None,
    )
    result.add_caption(pylatex.NoEscape(
        r"""
\roy{Plot of the \numEmissionLines\ brightest emission lines in the \ESIS\ passband.
Calculated using ChiantiPy, with the \cite{Schmelz2012} abundances, the \chiantiDEM\ \DEM\ file, and
$n_e T = $\,\chiantiPressure.}"""
    ))
    result.append(kgpy.latex.Label('fig:bunch'))
    return result
