import pathlib
import matplotlib.figure
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import kgpy.optics
import esis.optics
from . import formatting
from . import caching
from . import component_efficiency_vs_wavelength

__all__ = [
    'figure_mpl',
    'pdf',
]


def figure_mpl() -> matplotlib.figure.Figure:
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


def pdf() -> pathlib.Path:
    return caching.cache_pdf(figure_mpl)
