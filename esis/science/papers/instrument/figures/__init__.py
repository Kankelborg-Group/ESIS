import pathlib
import matplotlib.figure
import matplotlib.pyplot as plt
import astropy.units as u
import kgpy.vector
import esis.optics

__all__ = [
    'layout',
    'layout_pdf'
]

fig_width = 513.11743 / 72
column_width = 242.26653 / 72


def save_pdf(fig: matplotlib.figure.Figure, name: str) -> pathlib.Path:
    path = pathlib.Path(__file__).parent / (name + '.pdf')
    fig.savefig(
        fname=path,
        bbox_inches='tight',
        pad_inches=0.04,
    )
    plt.close(fig)
    return path


def layout() -> matplotlib.figure.Figure:
    fig_layout = plt.figure(figsize=(fig_width, fig_width))
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
    # esis_optics.primary.substrate_thickness = None
    # esis_optics.grating.substrate_thickness = None
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
    _, colorbar = esis_optics_rays.system.plot(
        ax=ax_layout,
        components=('y', 'z'),
        component_z='x',
        # plot_rays=False,
        plot_annotations=False,
        # plot_vignetted=True,
        plot_colorbar=False,
        plot_kwargs=dict(
            linewidth=0.5,
        ),
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


def layout_pdf() -> pathlib.Path:
    fig = layout()
    path = pathlib.Path(__file__).parent / 'layout_mpl.pdf'
    h = 1.5
    offset = (fig_width - h) / 2
    fig.savefig(
        fname=path,
        bbox_inches=fig.bbox_inches.from_bounds(0, offset, fig_width, h)
    )
    plt.close(fig)
    return path


def schematic() -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(fig_width, 5), constrained_layout=True)
    # fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)
    ax.margins(x=.01, y=.01)
    # ax.autoscale(enable=True, axis='both', tight=True)

    ax.set_aspect('equal')
    ax.set_axis_off()
    optics = esis.optics.design.final(all_channels=False)
    obs = optics.central_obscuration
    lines, colorbar = optics.system.plot(
        ax=ax,
        components=('z', 'x'),
        plot_rays=False,
        plot_annotations=False,
        annotation_text_y=2,
        plot_kwargs=dict(
            linewidth=1.5,
            solid_joinstyle='miter',
        ),
        # surface_first=optics.central_obscuration.surface,
    )
    optics.plot_distance_annotations_zx(ax=ax)

    ax.axhline(linestyle=(0, (20, 10)), linewidth=0.4, color='gray')
    ax.text(
        x=-500,
        y=0,
        s='axis of symmetry',
        ha='center',
        va='bottom',
        fontsize='small'
    )

    xh = kgpy.vector.x_hat.zx
    zh = kgpy.vector.z_hat.zx
    obs_zx = obs.transform.translation_eff.zx - obs.obscured_half_width * xh
    default_offset_x = -150 * u.mm
    apkw = dict(
        arrowstyle='->',
        linewidth=0.75,
        relpos=(0.5, 0.5),
    )
    kwargs_annotate = dict(
        fontsize='small',
        ha='center',
    )
    ax.annotate(
        text=str(obs.name),
        xy=obs_zx.to_tuple(),
        xytext=(obs_zx.x, default_offset_x),
        arrowprops=dict(
            # relpos=(0.5, 0.5),
            # connectionstyle='arc,angleA=90,angleB=-90,armA=10,armB=10',
            **apkw,
        ),
        # ha='center',
        **kwargs_annotate
    )
    width = optics.grating.inner_half_width + optics.grating.inner_border_width
    grating_z = optics.grating.substrate_thickness / 2 * zh
    grating_zx = optics.grating.transform.translation_eff.zx - (width * xh) - grating_z
    ax.annotate(
        text=str(optics.grating.name),
        xy=grating_zx.to_tuple(),
        xytext=(grating_zx.x + 100 * u.mm, default_offset_x),
        arrowprops=dict(
            connectionstyle='arc,angleA=90,angleB=-90,armA=15,armB=15',
            **apkw,
        ),
        **kwargs_annotate,

    )
    fs_zx = optics.field_stop.transform.translation_eff.zx
    ax.annotate(
        text=str(optics.field_stop.name),
        xy=fs_zx.to_tuple(),
        xytext=(fs_zx.x, default_offset_x),
        arrowprops=dict(
            **apkw,
        ),
        **kwargs_annotate,
    )
    primary_z = optics.primary.substrate_thickness / 2 * zh
    primary_zx = optics.primary.transform.translation_eff.zx - optics.primary.mech_half_width * xh + primary_z
    ax.annotate(
        text=str(optics.primary.name),
        xy=primary_zx.to_tuple(),
        xytext=(primary_zx.x, default_offset_x),
        arrowprops=dict(
            **apkw,
        ),
        **kwargs_annotate,
    )
    filter_zx = optics.filter.transform.translation_eff.zx - optics.filter.clear_radius * xh
    ax.annotate(
        text=str(optics.filter.name),
        xy=filter_zx.to_tuple(),
        xytext=(filter_zx.x, default_offset_x),
        arrowprops=dict(
            connectionstyle='arc,angleA=90,angleB=-90,armA=20,armB=30',
            **apkw,
        ),
        **kwargs_annotate,
    )
    detector_zx = optics.detector.transform.translation_eff.zx - optics.detector.clear_half_width * xh
    ax.annotate(
        text='detector',
        xy=detector_zx.to_tuple(),
        xytext=(detector_zx.x, default_offset_x),
        arrowprops=dict(
            connectionstyle='arc,angleA=90,angleB=-90,armA=20,armB=35',
            **apkw,
        ),
        **kwargs_annotate,
    )

    ax.set_ylabel(None)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(right=250)
    # colorbar.remove()

    return fig


def schematic_pdf() -> pathlib.Path:
    return save_pdf(schematic(), 'schematic')


def bunch(optics: esis.optics.Optics, digits_after_decimal: int) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(fig_width, 2), constrained_layout=True)
    optics.bunch.plot(ax=ax, num_emission_lines=optics.num_emission_lines, digits_after_decimal=digits_after_decimal)
    return fig


def bunch_pdf(optics: esis.optics.Optics, digits_after_decimal: int) -> pathlib.Path:
    fig = bunch(optics=optics, digits_after_decimal=digits_after_decimal)
    path = pathlib.Path(__file__).parent / 'bunch.pdf'
    fig.savefig(
        fname=path
    )
    plt.close(fig)
    return path


def field_stop_projections(optics: esis.optics.Optics, digits_after_decimal: int) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(column_width, 2.75), constrained_layout=True)
    optics.plot_field_stop_projections_local(ax=ax, digits_after_decimal=digits_after_decimal)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(0.5, -0.25), loc='upper center', ncol=2)
    return fig


def field_stop_projections_pdf(optics: esis.optics.Optics, digits_after_decimal: int) -> pathlib.Path:
    fig = field_stop_projections(optics=optics, digits_after_decimal=digits_after_decimal)
    return save_pdf(fig, 'field_stop_projections')
