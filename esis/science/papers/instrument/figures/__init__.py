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
import kgpy.optics
import esis.optics

__all__ = [
    'layout',
    'layout_pdf'
]

text_width = 513.11743 / 72
column_width = 242.26653 / 72

digits_after_decimal = 2
digits_after_decimal_schematic = 1


def save_pdf(fig_factory: typ.Callable[[], matplotlib.figure.Figure]) -> pathlib.Path:
    path = pathlib.Path(__file__).parent / (fig_factory.__name__ + '.pdf')
    if not path.exists():
        fig = fig_factory()
        fig.savefig(
            fname=path,
            # bbox_inches='tight',
            # pad_inches=0.04,
            # facecolor='lightblue'
        )
        plt.close(fig)
    return path


def layout() -> matplotlib.figure.Figure:
    fig_layout = plt.figure(figsize=(text_width, text_width))
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
    path = pathlib.Path(__file__).parent / 'layout_mpl.pdf'
    if not path.exists():
        fig = layout()
        h = 1.5
        offset = (text_width - h) / 2
        fig.savefig(
            fname=path,
            bbox_inches=fig.bbox_inches.from_bounds(0, offset, text_width, h)
        )
        plt.close(fig)
    return path


def schematic() -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(text_width, 2), constrained_layout=True)
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
    optics.plot_distance_annotations_zx(ax=ax, digits_after_decimal=digits_after_decimal_schematic,)

    ax.axhline(linestyle=(0, (20, 10)), linewidth=0.4, color='gray')
    ax.text(
        x=-500,
        y=0,
        s='axis of symmetry',
        ha='center',
        va='bottom',
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
    grating_z = -optics.grating.material.thickness / 2 * zh
    grating_zx = optics.grating.transform.translation_eff.zx - (width * xh) - grating_z
    ax.annotate(
        text=str(optics.grating.name),
        xy=grating_zx.to_tuple(),
        xytext=(grating_zx.x + 150 * u.mm, default_offset_x),
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
        xytext=(filter_zx.x - 100 * u.mm, default_offset_x),
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
        xytext=(detector_zx.x + 100 * u.mm, default_offset_x),
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
    return save_pdf(schematic)


def schematic_primary_and_obscuration() -> matplotlib.figure.Figure:

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
        fig, ax = plt.subplots(figsize=(column_width, 3.5), constrained_layout=True)
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
            digits_after_decimal=digits_after_decimal_schematic,
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
            digits_after_decimal=digits_after_decimal_schematic,
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
            digits_after_decimal=digits_after_decimal_schematic,
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
            points = points[index, mask[index]]
            hull = scipy.spatial.ConvexHull(points.xy.quantity)
            vertices = optics.system.transform_all(points[hull.vertices])
            if optics.grating.plot_kwargs['linestyle'][i] is None:
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


def schematic_primary_and_obscuration_pdf() -> pathlib.Path:
    return save_pdf(schematic_primary_and_obscuration)


def schematic_grating() -> matplotlib.figure.Figure:
    optics = esis.optics.design.final(all_channels=False)
    grating = optics.grating

    with astropy.visualization.quantity_support():
        fig, ax = plt.subplots(figsize=(column_width, 3.5), constrained_layout=True)
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
            digits_after_decimal=digits_after_decimal_schematic,
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
            digits_after_decimal=digits_after_decimal_schematic,
        )
        kgpy.plot.annotate_component(
            ax=ax,
            point_1=kgpy.vector.Vector2D(-grating.inner_half_width, -grating.width_short / 2),
            point_2=kgpy.vector.Vector2D(-grating.inner_half_width, grating.width_short / 2),
            component='y',
            position_orthogonal=-0.5 * grating.inner_half_width.value,
            digits_after_decimal=digits_after_decimal_schematic,
        )
        kgpy.plot.annotate_component(
            ax=ax,
            point_1=kgpy.vector.Vector2D(-grating.inner_half_width, grating.width_short / 2),
            point_2=kgpy.vector.Vector2D(grating.outer_half_width, grating.width_long / 2),
            position_orthogonal=1 * grating.width_mech_long.value / 2,
            digits_after_decimal=digits_after_decimal_schematic,
        )

        ax.text(
            x=0.3 * grating.outer_half_width.value,
            y=0,
            s='grating\nclear aperture',
            ha='center',
            va='center',
        )

    return fig


def schematic_grating_pdf() -> pathlib.Path:
    return save_pdf(schematic_grating)


def bunch() -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(text_width, 2), constrained_layout=True)
    optics = esis.optics.design.final()
    optics.bunch.plot(
        ax=ax,
        num_emission_lines=optics.num_emission_lines,
        digits_after_decimal=digits_after_decimal,
        label_fontsize=6,
    )
    return fig


def bunch_pdf() -> pathlib.Path:
    return save_pdf(bunch)


def field_stop_projections() -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(column_width, 3), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=.15)
    # ax.margins(x=.01, y=.01)
    optics = esis.optics.design.final(all_channels=False)
    optics.plot_field_stop_projections_local(
        ax=ax,
        wavelength_color=[
            'darkviolet',
            'indigo',
            'blue',
            'dodgerblue',
            'cyan',
            'green',
            'chartreuse',
            'orange',
            'orangered',
            'red',
        ],
        digits_after_decimal=digits_after_decimal,
        use_latex=True,
    )
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(0.5, -0.25), loc='upper center', ncol=2)
    return fig


def field_stop_projections_pdf() -> pathlib.Path:
    return save_pdf(field_stop_projections)


psf_pupil_samples = 201
psf_field_samples = 4


def psf() -> matplotlib.figure.Figure:
    optics = esis.optics.design.final(
        pupil_samples=psf_pupil_samples,
        pupil_is_stratified_random=True,
        field_samples=psf_field_samples,
        all_channels=False,
    )
    optics.num_emission_lines = 1

    rays = optics.rays_output

    bins = rays.input_grid.pupil.num_samples_normalized.x // 2

    fig, axs = rays.plot_pupil_hist2d_vs_field(
        wavlen_index=0,
        norm=matplotlib.colors.PowerNorm(1 / 3),
        bins=bins,
        cmap='gray_r',
        limits=((-0.5, 0.5), (-0.5, 0.5)),
    )
    fig.set_figheight(2.6)
    fig.set_figwidth(column_width)
    fig.suptitle(None)
    for i, axs_i in enumerate(axs):
        for j, axs_ij in enumerate(axs_i):
            ax = axs_ij
            if i + 1 == axs.shape[0]:
                ax.set_xlabel(None)
            if j == 0:
                ax.set_ylabel(None)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0, wspace=0)
    return fig


def psf_pdf() -> pathlib.Path:
    return save_pdf(psf)


kwargs_optics_default = dict(
    pupil_samples=21,
    pupil_is_stratified_random=True,
    field_samples=21,
    field_is_stratified_random=False,
    all_channels=False,
)

num_emission_lines_default = 3


def spot_size() -> matplotlib.figure.Figure:
    optics = esis.optics.design.final(**kwargs_optics_default)
    optics.num_emission_lines = num_emission_lines_default
    fig, axs = plt.subplots(
        ncols=optics.num_emission_lines,
        figsize=(text_width, 2.5),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )
    optics.rays_output.plot_spot_size_vs_field(
        axs=axs,
    )
    fullname = optics.bunch.fullname(
        digits_after_decimal=digits_after_decimal, use_latex=True)[np.argsort(optics.wavelength)]
    for name, ax in zip(fullname, axs):
        ax.set_title(name, fontsize=10)
    return fig


def spot_size_pdf() -> pathlib.Path:
    return save_pdf(spot_size)


def focus_curve() -> matplotlib.figure.Figure:
    optics = esis.optics.design.final(
        pupil_samples=11,
        pupil_is_stratified_random=True,
        field_samples=1,
        # field_is_stratified_random=True,
        all_channels=False,
    )
    optics.num_emission_lines = num_emission_lines_default
    fig, ax = plt.subplots(
        figsize=(column_width, 2.5),
        constrained_layout=True,
    )

    optics.plot_focus_curve(
        ax=ax,
        delta_detector=5 * u.mm,
        num_samples=51,
        digits_after_decimal=2,
        use_latex=True,
    )
    ax.legend(bbox_to_anchor=(0.5, -0.3), loc='upper center', ncol=2)
    fig.set_constrained_layout_pads(w_pad=.10, h_pad=0.10)

    return fig


def focus_curve_pdf() -> pathlib.Path:
    return save_pdf(focus_curve)


def vignetting() -> matplotlib.figure.Figure:
    optics = esis.optics.design.final(**kwargs_optics_default)
    fig, axs = plt.subplots(
        figsize=(column_width, 2.9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    model = optics.rays_output.vignetting(polynomial_degree=optics.vignetting_polynomial_degree)
    model.plot_unvignetted(axs=axs[0])
    axs[0, 0].set_title(
        optics.bunch.fullname(digits_after_decimal=digits_after_decimal, use_latex=True)[0], fontsize=10)
    return fig


def vignetting_pdf() -> pathlib.Path:
    return save_pdf(vignetting)


def distortion() -> matplotlib.figure.Figure:
    optics = esis.optics.design.final(**kwargs_optics_default)
    optics.num_emission_lines = 1
    fig, ax = plt.subplots(
        figsize=(column_width, 3.5),
        constrained_layout=True,
    )
    # fig.set_constrained_layout_pads(h_pad=.15)
    optics.plot_field_stop_distortion(
        ax=ax,
        digits_after_decimal=digits_after_decimal,
        use_latex=True,
    )
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center')
    return fig


def distortion_pdf() -> pathlib.Path:
    return save_pdf(distortion)


def distortion_residual() -> matplotlib.figure.Figure:
    optics = esis.optics.design.final(**kwargs_optics_default)
    optics.num_emission_lines = 3
    fig, axs = plt.subplots(
        nrows=2,
        ncols=optics.num_emission_lines,
        sharex=True,
        sharey=True,
        figsize=(text_width, 4.4),
        constrained_layout=True,
    )
    distortion_linear = optics.rays_output.distortion(polynomial_degree=1)
    distortion_quadratic = optics.rays_output.distortion(polynomial_degree=2)

    distortion_linear.plot_residual(
        axs=axs[0],
        use_xlabels=False,
    )
    distortion_quadratic.plot_residual(
        axs=axs[1],
        use_titles=False,
    )
    fullname = optics.bunch.fullname(
        digits_after_decimal=digits_after_decimal, use_latex=True)[np.argsort(optics.wavelength)]
    for name, ax in zip(fullname, axs[0]):
        ax.set_title(name, fontsize=10)

    return fig


def distortion_residual_pdf() -> pathlib.Path:
    return save_pdf(distortion_residual)


def grating_multilayer_schematic() -> matplotlib.figure.Figure:
    optics = esis.optics.design.final(**kwargs_optics_default)
    fig, ax = plt.subplots(
        figsize=(column_width, 2.2),
        constrained_layout=True,
    )
    optics.grating.surface.material.plot_layers(
        ax=ax,
        layer_material_color=dict(
            Al='lightblue',
            Mg='pink',
            SiC='lightgray',
        ),
        layer_label_x=dict(
            Al=1.0,
            Mg=0.5,
            SiC=0.5,
        ),
        layer_label_x_text=dict(
            Al=1.2,
            Mg=0.5,
            SiC=0.5,
        )
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    fig.set_constrained_layout_pads(w_pad=0.2)
    return fig


def grating_multilayer_schematic_pdf() -> pathlib.Path:
    return save_pdf(grating_multilayer_schematic)


def grating_efficiency_vs_angle() -> matplotlib.figure.Figure:
    with astropy.visualization.quantity_support():
        optics = esis.optics.design.final(**kwargs_optics_default)
        fig, ax = plt.subplots(
            figsize=(column_width, 2),
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
            x=angle_m0,
            y=1.01,
            s='$m=0$',
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='bottom',
        )
        ax.text(
            x=angle_m1,
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


def grating_efficiency_vs_angle_pdf() -> pathlib.Path:
    return save_pdf(grating_efficiency_vs_angle)


def _annotate_wavelength(
        ax: matplotlib.axes.Axes,
        label_orders: bool = True
) -> typ.List[matplotlib.lines.Line2D]:
    optics = esis.optics.design.final(**kwargs_optics_default)
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
        digits_after_decimal=digits_after_decimal,
        colors=['orangered', 'blue'],
    )
    lines += optics2.bunch.plot_wavelength(
        ax=ax,
        num_emission_lines=optics2.num_emission_lines,
        digits_after_decimal=digits_after_decimal,
        colors=['gray', 'black'],
    )
    return lines


def grating_efficiency_vs_wavelength() -> matplotlib.figure.Figure:
    fig, axs = plt.subplots(
        nrows=2,
        sharex=True,
        figsize=(column_width, 5),
        constrained_layout=True,
    )
    eff_unit = u.percent
    wavl_unit = u.Angstrom
    witness = esis.optics.grating.efficiency.witness
    with astropy.visualization.quantity_support():
        for func in [witness.vs_wavelength_g17, witness.vs_wavelength_g19, witness.vs_wavelength_g24]:
            serial, angle_input, wavelength, efficiency = func()
            axs[0].plot(
                wavelength.to(wavl_unit),
                efficiency.to(eff_unit),
                label=serial,
            )
        axs[0].legend()

        axs[0].set_xlabel(None)
        axs[0].set_ylabel(f'efficiency ({eff_unit:latex})')
        _annotate_wavelength(ax=axs[0])

        angle_input, wavelength, efficiency = esis.optics.grating.efficiency.vs_wavelength()
        axs[1].plot(
            wavelength.to(wavl_unit),
            efficiency.to(eff_unit),
            label='grating 017',
        )
        axs[1].add_artist(axs[1].legend())
        lines = _annotate_wavelength(ax=axs[1], label_orders=False)
        axs[1].set_xlabel(f'wavelength ({wavl_unit:latex})')
        axs[1].set_ylabel(f'efficiency ({eff_unit:latex})')
        axs[1].legend(handles=lines, bbox_to_anchor=(0.5, -0.25), loc='upper center', ncol=2)

    return fig


def grating_efficiency_vs_wavelength_pdf() -> pathlib.Path:
    return save_pdf(grating_efficiency_vs_wavelength)


def grating_efficiency_vs_position() -> matplotlib.figure.Figure:
    with astropy.visualization.quantity_support():
        fig, axs = plt.subplots(
            ncols=2,
            sharey=True,
            figsize=(text_width, 2.5),
            constrained_layout=True,
        )

        position_x, position_y, wavelength, efficiency = esis.optics.grating.efficiency.vs_position_x()
        axs[0].plot(
            position_x,
            efficiency,
            label='grating 017',
        )
        axs[0].set_xlabel(f'$x$ position ({axs[0].get_xlabel()})')
        axs[0].set_ylabel(f'efficiency ({axs[0].get_ylabel()})')
        axs[0].legend()

        position_x, position_y, wavelength, efficiency = esis.optics.grating.efficiency.vs_position_y()
        axs[1].plot(
            position_y,
            efficiency,
            label='grating 017'
        )
        axs[1].set_xlabel(f'$y$ position ({axs[1].get_xlabel()})')
        axs[1].set_ylabel(None)
        axs[1].legend()

        return fig


def grating_efficiency_vs_position_pdf() -> pathlib.Path:
    return save_pdf(grating_efficiency_vs_position)


def primary_efficiency_vs_wavelength() -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(
        figsize=(column_width, 2.75),
        constrained_layout=True,
    )
    eff_unit = u.percent
    wavl_unit = u.Angstrom
    witness = esis.optics.primary.efficiency.witness
    with astropy.visualization.quantity_support():
        for func in [witness.vs_wavelength_p1, witness.vs_wavelength_p2,]:
            serial, angle_input, wavelength, efficiency = func()
            ax.plot(
                wavelength.to(wavl_unit),
                efficiency.to(eff_unit),
                label=serial,
            )
        ax.legend()
        ax.add_artist(ax.legend())
        lines = _annotate_wavelength(ax=ax)
        ax.set_xlabel(f'wavelength ({wavl_unit:latex})')
        ax.set_ylabel(f'reflectivity ({eff_unit:latex})')
        ax.legend(handles=lines, bbox_to_anchor=(0.5, -0.25), loc='upper center', ncol=2)

    return fig


def primary_efficiency_vs_wavelength_pdf() -> pathlib.Path:
    return save_pdf(primary_efficiency_vs_wavelength)


def filter_efficiency_vs_wavelength() -> matplotlib.figure.Figure:
    with astropy.visualization.quantity_support():
        fig, ax = plt.subplots(
            figsize=(column_width, 2.75),
            constrained_layout=True,
        )
        optics = esis.optics.design.final(**kwargs_optics_default)
        wavelength = esis.optics.grating.efficiency.witness.vs_wavelength_g17()[2].to(u.AA)
        rays = kgpy.optics.rays.Rays(
            wavelength=wavelength,
        )
        ax.plot(
            wavelength,
            optics.filter.surface.material.transmissivity(rays),
            label=r'$\mathrm{Al}_2\mathrm{O}_3 / \mathrm{Al} / \mathrm{Al}_2\mathrm{O}_3$',
        )
        ax.add_artist(ax.legend())
        lines = _annotate_wavelength(ax=ax)
        ax.set_xlabel(f'wavelength ({ax.get_xlabel()})')
        ax.set_ylabel(f'transmission ({ax.get_ylabel()})')
        # ax.legend()
        ax.legend(handles=lines, bbox_to_anchor=(0.5, -0.3), loc='upper center', ncol=2)

        fig.set_constrained_layout_pads(h_pad=0.1)
        return fig


def filter_efficiency_vs_wavelength_pdf() -> pathlib.Path:
    return save_pdf(filter_efficiency_vs_wavelength)