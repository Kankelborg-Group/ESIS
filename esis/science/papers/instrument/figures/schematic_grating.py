import pathlib
import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt
import scipy.spatial
import astropy.units as u
import astropy.visualization
import kgpy.vector
import kgpy.plot
import esis.optics
from . import formatting
from . import caching

__all__ = [
    'figure_mpl',
    'pdf',
]


def figure_mpl() -> matplotlib.figure.Figure:
    optics = esis.optics.design.final(all_channels=False)
    grating = optics.grating

    with astropy.visualization.quantity_support():
        fig, ax = plt.subplots(figsize=(formatting.column_width, 3.5), constrained_layout=True)
        ax.margins(x=.01, y=.01)
        ax.set_aspect('equal')
        ax.set_axis_off()

        vertices_grating_mech = optics.grating.surface.aperture_mechanical.vertices
        # vertices_grating_mech = optics.system.transform_all(vertices_primary_mech)
        ax.fill(
            vertices_grating_mech.x,
            vertices_grating_mech.y,
            facecolor='gray',
            edgecolor='black',
        )
        kgpy.plot.annotate_component(
            ax=ax,
            point_1=kgpy.vector.Vector2D(-grating.inner_half_width - grating.inner_border_width, grating.width_mech_short / 2),
            point_2=kgpy.vector.Vector2D(grating.outer_half_width + grating.border_width, grating.width_mech_long / 2),
            # point_2=kgpy.vector.Vector2D(grating.height_mech / 2, grating.width_mech_long / 2),
            position_orthogonal=1.2 * grating.width_mech_long.value / 2,
            digits_after_decimal=formatting.digits_after_decimal_schematic,
        )

        vertices_grating_clear = optics.grating.surface.aperture.vertices
        # vertices_grating_mech = optics.system.transform_all(vertices_primary_mech)
        ax.fill(
            vertices_grating_clear.x,
            vertices_grating_clear.y,
            # hatch='/',
            # fill=False,
            facecolor='white',
            edgecolor='black',
        )
        kgpy.plot.annotate_component(
            ax=ax,
            point_1=kgpy.vector.Vector2D(grating.outer_half_width, -grating.width_long / 2),
            point_2=kgpy.vector.Vector2D(grating.outer_half_width, grating.width_long / 2),
            component='y',
            position_orthogonal=1.3 * (grating.outer_half_width + grating.border_width).value,
            digits_after_decimal=formatting.digits_after_decimal_schematic,
        )
        kgpy.plot.annotate_component(
            ax=ax,
            point_1=kgpy.vector.Vector2D(-grating.inner_half_width, -grating.width_short / 2),
            point_2=kgpy.vector.Vector2D(-grating.inner_half_width, grating.width_short / 2),
            component='y',
            position_orthogonal=-0.5 * grating.inner_half_width.value,
            digits_after_decimal=formatting.digits_after_decimal_schematic,
        )
        kgpy.plot.annotate_component(
            ax=ax,
            point_1=kgpy.vector.Vector2D(-grating.inner_half_width, grating.width_short / 2),
            point_2=kgpy.vector.Vector2D(grating.outer_half_width, grating.width_long / 2),
            position_orthogonal=1 * grating.width_mech_long.value / 2,
            digits_after_decimal=formatting.digits_after_decimal_schematic,
        )

        ax.text(
            x=0.3 * grating.outer_half_width.value,
            y=0,
            s='grating\nclear aperture',
            ha='center',
            va='center',
        )

    return fig


def pdf() -> pathlib.Path:
    return caching.cache_pdf(figure_mpl)
