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
]


def distortion() -> matplotlib.figure.Figure:
    optics = esis.optics.design.final(**optics_factories.default_kwargs)
    optics.num_emission_lines = 1
    fig, ax = plt.subplots(
        figsize=(formatting.column_width, 3.5),
        constrained_layout=True,
    )
    # fig.set_constrained_layout_pads(h_pad=.15)
    optics.plot_field_stop_distortion(
        ax=ax,
        digits_after_decimal=formatting.digits_after_decimal,
        use_latex=True,
    )
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center')
    return fig


def distortion_pdf() -> pathlib.Path:
    return caching.cache_pdf(distortion)


def distortion_residual() -> matplotlib.figure.Figure:
    optics_quadratic = optics_factories.as_designed_single_channel()
    optics_linear = optics_quadratic.copy()
    optics_linear.distortion_polynomial_degree = 1
    optics_linear.update()
    fig, axs = plt.subplots(
        nrows=2,
        ncols=optics_factories.num_emission_lines_default,
        sharex=True,
        sharey=True,
        figsize=(formatting.text_width, 4.4),
        constrained_layout=True,
    )
    distortion_linear = optics_linear.rays_output.distortion
    distortion_quadratic = optics_quadratic.rays_output.distortion

    distortion_linear.plot_residual(
        axs=axs[0],
        use_xlabels=False,
        wavelength_name=optics_linear.bunch.fullname(formatting.digits_after_decimal),
    )
    distortion_quadratic.plot_residual(
        axs=axs[1],
        use_titles=False,
    )

    return fig


def distortion_residual_pdf() -> pathlib.Path:
    return caching.cache_pdf(distortion_residual)


def distortion_residual_relative() -> matplotlib.figure.Figure:
    optics_quadratic = optics_factories.as_designed_single_channel()
    optics_linear = optics_quadratic.copy()
    optics_linear.distortion_polynomial_degree = 1
    optics_linear.update()
    fig, axs = plt.subplots(
        nrows=2,
        ncols=optics_factories.num_emission_lines_default,
        sharex=True,
        sharey=True,
        figsize=(formatting.text_width, 4.4),
        constrained_layout=True,
    )
    distortion_linear = optics_linear.rays_output_relative.distortion
    distortion_quadratic = optics_quadratic.rays_output_relative.distortion

    distortion_linear.plot_residual(
        axs=axs[0],
        use_xlabels=False,
        wavelength_name=optics_linear.bunch.fullname(formatting.digits_after_decimal),
    )
    distortion_quadratic.plot_residual(
        axs=axs[1],
        use_titles=False,
    )

    return fig


def distortion_residual_relative_pdf() -> pathlib.Path:
    return caching.cache_pdf(distortion_residual_relative)


def grating_multilayer_schematic() -> matplotlib.figure.Figure:
    optics = esis.optics.design.final(**optics_factories.default_kwargs)
    fig, ax = plt.subplots(
        figsize=(formatting.column_width, 2.2),
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
    return caching.cache_pdf(grating_multilayer_schematic)


def grating_efficiency_vs_angle() -> matplotlib.figure.Figure:
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


def grating_efficiency_vs_angle_pdf() -> pathlib.Path:
    return caching.cache_pdf(grating_efficiency_vs_angle)


def _annotate_wavelength(
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


def component_efficiency_vs_wavelength() -> matplotlib.figure.Figure:
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
        _annotate_wavelength(ax=axs[0])

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
        lines = _annotate_wavelength(ax=axs[1], label_orders=False)
        axs[1].set_xlabel(f'wavelength ({wavl_unit:latex})')
        axs[1].set_ylabel(f'efficiency ({eff_unit:latex})')
        axs[1].legend(handles=lines, bbox_to_anchor=(0.5, -0.25), loc='upper center', ncol=2)

    return fig


def component_efficiency_vs_wavelength_pdf() -> pathlib.Path:
    return caching.cache_pdf(component_efficiency_vs_wavelength)


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


def primary_efficiency_vs_wavelength() -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(
        figsize=(formatting.column_width, 2.75),
        constrained_layout=True,
    )
    eff_unit = u.percent
    wavl_unit = u.Angstrom
    witness = esis.optics.primary.efficiency.witness
    with astropy.visualization.quantity_support():
        for func in [witness.vs_wavelength_p1, witness.vs_wavelength_p2, witness.vs_wavelength_recoat_1]:
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
    return caching.cache_pdf(primary_efficiency_vs_wavelength)


def filter_efficiency_vs_wavelength() -> matplotlib.figure.Figure:
    with astropy.visualization.quantity_support():
        fig, ax = plt.subplots(
            figsize=(formatting.column_width, 2.75),
            constrained_layout=True,
        )
        optics = esis.optics.design.final(**optics_factories.default_kwargs)
        wavelength = esis.optics.grating.efficiency.witness.vs_wavelength_g17()[2].to(u.AA)
        rays = kgpy.optics.rays.Rays(
            wavelength=wavelength,
        )
        ax.plot(
            wavelength,
            optics.filter.surface.material.transmissivity(rays).to(u.percent),
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
    return caching.cache_pdf(filter_efficiency_vs_wavelength)


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
        lines = _annotate_wavelength(ax=ax)
        ax.set_xlabel(f'wavelength ({ax.get_xlabel()})')
        ax.set_ylabel(f'quantum efficiency ({ax.get_ylabel()})')
        # ax.legend()
        ax.legend(handles=lines, bbox_to_anchor=(0.5, -0.3), loc='upper center', ncol=2)

        fig.set_constrained_layout_pads(h_pad=0.1)
        return fig


def ccd_efficiency_vs_wavelength_pdf() -> pathlib.Path:
    return caching.cache_pdf(ccd_efficiency_vs_wavelength)


def alignment_transfer() -> matplotlib.figure.Figure:

    fig, ax = plt.subplots(
        figsize=(formatting.column_width, 4),
        constrained_layout=True,
    )

    optics = esis.optics.design.final(**optics_factories.default_kwargs)
    grating = optics.grating
    grating.piston = 0 * u.mm
    grating.cylindrical_azimuth = 0 * u.deg
    grating.cylindrical_radius = 0 * u.mm
    grating.inclination = 0 * u.deg
    grating.diffraction_order = 0 * u.dimensionless_unscaled
    surface_4d = kgpy.optics.surface.Surface(
        name=kgpy.Name('diverger'),
        transform=kgpy.transform.rigid.TransformList([kgpy.transform.rigid.Translate(z=grating.sagittal_radius)]),
        aperture=kgpy.optics.surface.aperture.Circular(radius=10 * u.mm)
    )
    surface_4d_return = surface_4d.copy()
    surface_4d_return.name = kgpy.Name()

    red_hene = 632.8 * u.nm
    system_alignment = kgpy.optics.System(
        grid_wavelength=kgpy.grid.RegularGrid1D(
            min=red_hene,
            max=red_hene,
        ),
        object_surface=surface_4d,
        surfaces=kgpy.optics.surface.SurfaceList([
            grating.surface,
            surface_4d_return,
        ]),
        field_samples=1,
        field_margin=1 * u.nm,
    )
    system_alignment.plot(
        ax=ax,
        components=('z', 'x'),
        plot_vignetted=True,
        plot_colorbar=False,
    )

    grating.diffraction_order = [0, -1] * u.dimensionless_unscaled
    green_laser_wavelength = 532 * u.nm
    cylinder_azimuth = -40 * u.deg
    system_roll = kgpy.optics.System(
        grid_wavelength=kgpy.grid.RegularGrid1D(
            min=green_laser_wavelength,
            max=green_laser_wavelength,
        ),
        object_surface=kgpy.optics.surface.Surface(
            name=kgpy.Name('cylinder'),
            transform=kgpy.transform.rigid.TransformList([
                kgpy.transform.rigid.TiltY(cylinder_azimuth),
                kgpy.transform.rigid.Translate(z=grating.sagittal_radius)
            ]),
            aperture=kgpy.optics.surface.aperture.Rectangular(
                half_width_x=7 * u.mm,
                half_width_y=14 * u.mm,
            )
        ),
        surfaces=kgpy.optics.surface.SurfaceList([
            grating.surface,
            kgpy.optics.surface.Surface(
                name=kgpy.Name('screen'),
                transform=kgpy.transform.rigid.TransformList([
                    kgpy.transform.rigid.TiltY(u.Quantity([-cylinder_azimuth, cylinder_azimuth])),
                    kgpy.transform.rigid.Translate(z=grating.sagittal_radius)
                ]),
                aperture=kgpy.optics.surface.aperture.Rectangular(
                    half_width_x=5 * u.imperial.inch,
                    half_width_y=3 * u.imperial.inch,
                )
            )
        ]),
        field_samples=1,
        field_margin=1 * u.nm,
    )
    system_roll.plot(
        ax=ax,
        components=('z', 'x'),
        plot_vignetted=True,
        # plot_rays=False,
        plot_colorbar=False,
    )

    ax.set_aspect('equal')

    return fig


def alignment_transfer_pdf() -> pathlib.Path:
    return caching.cache_pdf(alignment_transfer)
