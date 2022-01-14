import typing as typ
import pathlib
import matplotlib.figure
import matplotlib.colors
import matplotlib.lines
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
import astropy.units as u
import astropy.visualization
import kgpy.vector
import kgpy.format
import kgpy.plot
import kgpy.transform
import kgpy.optics
import kgpy.grid
import esis.optics
from .. import optics as optics_factories
from . import formatting
from . import caching
from . import schematic_moses
from . import layout
from . import bunch
from . import schematic
from . import schematic_primary
from . import schematic_grating
from . import field_stop_projections
from . import psf
from . import spot_size
from . import focus_curve
from . import vignetting
from . import distortion
from . import distortion_residual
from . import grating_multilayer_schematic
from . import grating_efficiency_vs_angle
from . import component_efficiency_vs_wavelength

__all__ = [
    'schematic_moses',
    'layout',
    'bunch',
    'schematic',
    'schematic_primary',
    'schematic_grating',
    'field_stop_projections',
    'psf',
    'spot_size',
    'focus_curve',
    'vignetting',
    'distortion',
    'distortion_residual',
    'grating_multilayer_schematic',
    'grating_efficiency_vs_angle',
    'component_efficiency_vs_wavelength',
]


def grating_efficiency_vs_position() -> matplotlib.figure.Figure:
    with astropy.visualization.quantity_support():
        fig, axs = plt.subplots(
            ncols=2,
            sharey=True,
            figsize=(formatting.text_width, 2.5),
            constrained_layout=True,
        )

        position_x, position_y, wavelength, efficiency = esis.optics.grating.efficiency.vs_position_x()
        axs[0].plot(
            position_x,
            efficiency.to(u.percent),
            label='grating 017',
        )
        axs[0].set_xlabel(f'$x$ position ({axs[0].get_xlabel()})')
        axs[0].set_ylabel(f'efficiency ({axs[0].get_ylabel()})')

        position_x, position_y, wavelength, efficiency = esis.optics.grating.efficiency.vs_position_y()
        axs[1].plot(
            position_y,
            efficiency,
            label='grating 017'
        )
        axs[1].set_xlabel(f'$y$ position ({axs[1].get_xlabel()})')
        axs[1].set_ylabel(None)

        return fig


def grating_efficiency_vs_position_pdf() -> pathlib.Path:
    return caching.cache_pdf(grating_efficiency_vs_position)


def ccd_efficiency_vs_wavelength() -> matplotlib.figure.Figure:
    with astropy.visualization.quantity_support():
        fig, ax = plt.subplots(
            figsize=(formatting.column_width, 2.75),
            constrained_layout=True,
        )
        wavelength = esis.optics.grating.efficiency.witness.vs_wavelength_g17()[2].to(u.AA)
        rays = kgpy.optics.rays.Rays(
            wavelength=wavelength,
        )
        ax.plot(
            wavelength,
            # wavelength,
            kgpy.optics.surface.material.CCDStern1994().transmissivity(rays).to(u.percent),
            label=r'Stern 1994',
        )
        ax.plot(
            wavelength,
            # wavelength,
            kgpy.optics.surface.material.CCDStern2004().transmissivity(rays),
            label=r'Stern 2004',
        )
        ax.add_artist(ax.legend())
        lines = component_efficiency_vs_wavelength.annotate_wavelength(ax=ax)
        ax.set_xlabel(f'wavelength ({ax.get_xlabel()})')
        ax.set_ylabel(f'quantum efficiency ({ax.get_ylabel()})')
        # ax.legend()
        ax.legend(handles=lines, bbox_to_anchor=(0.5, -0.3), loc='upper center', ncol=2)

        fig.set_constrained_layout_pads(h_pad=0.1)
        return fig


def ccd_efficiency_vs_wavelength_pdf() -> pathlib.Path:
    return caching.cache_pdf(ccd_efficiency_vs_wavelength)
