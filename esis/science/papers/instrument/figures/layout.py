import pathlib
import matplotlib.figure
import matplotlib.pyplot as plt
import astropy.units as u
import pylatex
import kgpy.latex
import kgpy.vector
import esis.optics
from . import formatting
from . import caching

__all__ = []


def figure_mpl() -> matplotlib.figure.Figure:
    fig_layout = plt.figure(figsize=(formatting.text_width, formatting.text_width))
    ax_layout = fig_layout.add_subplot(111, projection='3d')
    # fig_layout, ax_layout = plt.subplots(figsize=(7.1, 4), constrained_layout=True)
    ax_layout.set_axis_off()
    # ax_layout.set_aspect('equal')
    # ax_layout.set_proj_type('ortho')
    esis_optics = esis.optics.design.final(pupil_samples=kgpy.vector.Vector2D(3, 1), field_samples=1)
    # esis_optics.pointing.x = 3 * u.deg
    # esis_optics.pointing.y = 60 * u.deg
    esis_optics.central_obscuration = None
    esis_optics.filter = None
    esis_optics.source.piston = 1425 * u.mm
    esis_optics.front_aperture.piston = esis_optics.source.piston

    # print()

    esis_optics_rays = esis_optics.copy()
    chan_index = 4
    esis_optics_rays.grating.cylindrical_azimuth = esis_optics_rays.grating.cylindrical_azimuth[chan_index]
    # esis_optics_rays.filter.cylindrical_azimuth = esis_optics_rays.filter.cylindrical_azimuth[chan_index]
    esis_optics_rays.detector.cylindrical_azimuth = esis_optics_rays.detector.cylindrical_azimuth[chan_index]
    esis_optics_rays.grating.plot_kwargs['linestyle'] = esis_optics_rays.grating.plot_kwargs['linestyle'][chan_index]
    esis_optics_rays.detector.plot_kwargs['linestyle'] = esis_optics_rays.detector.plot_kwargs['linestyle'][chan_index]
    esis_optics_rays.grating.plot_kwargs['alpha'] = esis_optics_rays.grating.plot_kwargs['alpha'][chan_index]
    esis_optics_rays.detector.plot_kwargs['alpha'] = esis_optics_rays.detector.plot_kwargs['alpha'][chan_index]
    esis_optics_rays.num_emission_lines = 1

    esis_optics.system.plot(
        ax=ax_layout,
        components=('y', 'z'),
        component_z='x',
        plot_rays=False,
        plot_annotations=False,
        plot_kwargs=dict(
            linewidth=0.5,
        ),
    )

    index_field_stop = esis_optics_rays.system.surfaces_all.flat_local.index(esis_optics_rays.field_stop.surface)
    esis_optics_rays.system.raytrace[index_field_stop].vignetted_mask = esis_optics_rays.system.raytrace[~0].vignetted_mask
    _, colorbar = esis_optics_rays.system.raytrace[:index_field_stop + 1].plot(
        ax=ax_layout,
        components=('y', 'z'),
        component_z='x',
        plot_colorbar=False,
        plot_kwargs=dict(
            linewidth=0.5,
            zorder=25,
            color='tab:blue',
        ),
    )
    _, colorbar = esis_optics_rays.system.raytrace[index_field_stop:].plot(
        ax=ax_layout,
        components=('y', 'z'),
        component_z='x',
        plot_colorbar=False,
        plot_kwargs=dict(
            linewidth=0.5,
            zorder=30,
            color='tab:blue',
        ),
    )

    ax_layout.text(
        x=0,
        y=esis_optics.primary.translation.z.value + 20,
        z=esis_optics.primary.mech_half_width.value + 15,
        s='primary mirror',
        ha='right',
        va='bottom',
    )

    ax_layout.text(
        x=0,
        y=esis_optics.detector.translation.z.value + 20,
        z=-(esis_optics.detector.cylindrical_radius + esis_optics.detector.clear_half_width).to(u.mm).value - 20,
        s='detectors',
        ha='center',
        va='center',
    )

    ax_layout.text(
        x=0,
        y=-esis_optics.field_stop.piston.value,
        z=-(esis_optics.field_stop.mech_radius).to(u.mm).value - 5,
        s='field stop',
        ha='center',
        va='top',
    )
    ax_layout.text(
        x=0,
        y=esis_optics.grating.translation.z.value - 50,
        z=-(esis_optics.grating.cylindrical_radius + esis_optics.grating.outer_half_width).to(u.mm).value - 15,
        s='diffraction gratings',
        ha='left',
        va='top',
    )

    ax_layout.view_init(elev=0, azim=-40)
    # ax_layout.autoscale(tight=True)
    # ax_layout.dist = 8

    xlim = ax_layout.get_xlim()
    ylim = ax_layout.get_ylim()
    zlim = ax_layout.get_zlim()
    ax_layout.set_box_aspect(
        (
            (xlim[1] - xlim[0]),
            (ylim[1] - ylim[0]),
            (zlim[1] - zlim[0]),
        ),
        zoom=1.3,
    )
    # ax_layout.set_xlim3d([-1/2, 1/2])
    # ax_layout.set_ylim3d([-1/2, 1/2])
    # ax_layout.set_zlim3d([-2, 2])
    # ax_layout.set_aspect('auto')

    fig_layout.subplots_adjust(left=0, right=1, bottom=0, top=1)

    return fig_layout


def pdf() -> pathlib.Path:
    path = pathlib.Path(__file__).parent / 'output' / 'layout.pdf'
    if not path.exists():
        fig = figure_mpl()
        h = 1.45
        offset = (formatting.text_width - h) / 2
        fig.savefig(
            fname=path,
            bbox_inches=fig.bbox_inches.from_bounds(0, 3, formatting.text_width, h),
        )
        plt.close(fig)
    return path


def figure() -> pylatex.Figure:
    result = pylatex.Figure(position='!ht')
    result._star_latex_name = True
    result.add_image(
        filename=str(pdf()),
        width=None,
    )
    result.add_caption(pylatex.NoEscape(
        r"""
\roy{\ESIS\ optical layout. 
Dashed lines indicate the positions of unpopulated channels. 
The blue lines represent the path of \OV\ through the system.} The \ESIS\ instrument is a pseudo-Gregorian design.
The secondary mirror is replaced by a segmented array of concave diffraction gratings.
The field stop at prime focus defines instrument spatial/spectral \FOV.
\CCDs\ are arrayed around the primary mirror, each associated with a particular grating.
Eight grating positions appear in this schematic; only six fit within the volume of the rocket payload.
\NumChannelsWords\ channels are populated for the first flight."""
    ))
    result.append(kgpy.latex.Label('fig:layout'))
    return result