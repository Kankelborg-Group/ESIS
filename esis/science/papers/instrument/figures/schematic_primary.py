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

    optics = esis.optics.design.final(
        pupil_samples=21,
        pupil_is_stratified_random=True,
        field_samples=7,
        field_is_stratified_random=True,
    )
    optics.roll = -optics.detector.cylindrical_azimuth[1]
    primary = optics.primary
    obscuration = optics.central_obscuration

    with astropy.visualization.quantity_support():
        fig, ax = plt.subplots(figsize=(formatting.column_width, 3.5), constrained_layout=True)
        ax.margins(x=.01, y=.01)
        ax.set_aspect('equal')
        ax.set_axis_off()
        kwargs_plot = dict(
            ax=ax,
            transform_extra=optics.system.transform_all,
            plot_annotations=False,
        )
        # optics.primary.plot(**kwargs_plot)

        vertices_primary_mech = primary.surface.aperture_mechanical.vertices
        vertices_primary_mech = optics.system.transform_all(vertices_primary_mech)
        ax.fill(
            vertices_primary_mech.x,
            vertices_primary_mech.y,
            facecolor='gray',
            edgecolor='black',
        )
        kgpy.plot.annotate_component(
            ax=ax,
            point_1=kgpy.vector.Vector2D.from_cylindrical(primary.mech_radius, 22.5 * u.deg),
            point_2=kgpy.vector.Vector2D.from_cylindrical(primary.mech_radius, (180 - 22.5) * u.deg),
            position_orthogonal=1.6 * primary.mech_half_width.value,
            digits_after_decimal=formatting.digits_after_decimal_schematic,
        )

        vertices_primary = primary.surface.aperture.vertices
        vertices_primary = optics.system.transform_all(vertices_primary)
        ax.fill(
            vertices_primary.x,
            vertices_primary.y,
            facecolor='white',
            edgecolor='black',
        )
        kgpy.plot.annotate_component(
            ax=ax,
            point_1=kgpy.vector.Vector2D.from_cylindrical(primary.clear_radius, 22.5 * u.deg),
            point_2=kgpy.vector.Vector2D.from_cylindrical(primary.clear_radius, (180 - 22.5) * u.deg),
            position_orthogonal=1.4 * primary.mech_half_width.value,
            digits_after_decimal=formatting.digits_after_decimal_schematic,
        )
        ax.text(
            x=0,
            y=(obscuration.obscured_half_width + primary.mech_half_width) / 2,
            s='primary\nclear aperture',
            ha='center',
            va='center',
        )


        vertices_obscuration = obscuration.surface.aperture.vertices
        vertices_obscuration = optics.system.transform_all(vertices_obscuration)
        ax.fill(
            vertices_obscuration.x,
            vertices_obscuration.y,
            facecolor='darkgray',
            edgecolor='black',
        )
        kgpy.plot.annotate_component(
            ax=ax,
            point_1=kgpy.vector.Vector2D.from_cylindrical(obscuration.obscured_radius, 22.5 * u.deg),
            point_2=kgpy.vector.Vector2D.from_cylindrical(obscuration.obscured_radius, (180 - 22.5) * u.deg),
            position_orthogonal=1.2 * primary.mech_half_width.value,
            digits_after_decimal=formatting.digits_after_decimal_schematic,
        )
        ax.text(
            x=0,
            y=0,
            s='central\nobscuration',
            ha='center',
            va='center',
        )

        rays = optics.system.raytrace[optics.system.surfaces_all.flat_local.index(primary.surface)]
        mask = optics.system.rays_output.mask
        for i in range(rays.size):
            index = np.unravel_index(i, rays.shape)
            points = np.broadcast_to(rays.position, mask.shape, subok=True)
            points = points[index][mask[index]]
            hull = scipy.spatial.ConvexHull(points.xy.quantity)
            vertices = optics.system.transform_all(points[hull.vertices])
            if optics.grating.plot_kwargs['linestyle'][i] is 'solid':
                alpha = 1.0
                inactive = ''
            else:
                alpha = 0.5
                inactive = '\n(inactive)'
            ax.fill(
                vertices.x,
                vertices.y,
                fill=False,
                edgecolor='red',
                alpha=alpha,
            )
            position_label_channel = kgpy.vector.Vector2D.from_cylindrical(
                radius=(obscuration.obscured_half_width + primary.mech_half_width) / 2,
                azimuth=optics.grating.cylindrical_azimuth[i] + 180 * u.deg,
            )
            position_label_channel = optics.system.transform_all(position_label_channel.to_3d()).xy
            ax.text(
                x=position_label_channel.x.value,
                y=position_label_channel.y.value,
                s='Channel {}{}'.format(i, inactive),
                ha='center',
                va='center',
                color='red',
                alpha=alpha,
            )

    return fig


def pdf() -> pathlib.Path:
    return caching.cache_pdf(figure_mpl)
