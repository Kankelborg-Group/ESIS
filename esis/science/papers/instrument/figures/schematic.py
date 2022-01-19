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
from . import schematic_primary
from . import schematic_grating

__all__ = [
    'figure_mpl',
]


def figure_mpl() -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(formatting.text_width, 2), constrained_layout=True)
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
    optics.plot_distance_annotations_zx(ax=ax, digits_after_decimal=formatting.digits_after_decimal_schematic,)

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
    primary_z = optics.primary.material.thickness / 2 * zh
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


def pdf() -> pathlib.Path:
    return caching.cache_pdf(figure_mpl)


def figure() -> pylatex.Figure:
    result = pylatex.Figure(position='htb!')
    result._star_latex_name = True
    result.append(kgpy.latex.aas.Gridline([
        kgpy.latex.aas.Fig(pdf(), kgpy.latex.textwidth, '(a)')
    ]))

    result.append(kgpy.latex.aas.Gridline([
        kgpy.latex.aas.LeftFig(schematic_primary.pdf(), kgpy.latex.columnwidth, '(b)'),
        kgpy.latex.aas.RightFig(schematic_grating.pdf(), kgpy.latex.columnwidth, '(c)'),
    ]))

    result.add_caption(pylatex.NoEscape(
        r"""(a) Schematic diagram of a single channel of the \ESIS\ optical system.
(b) Clear aperture of the primary mirror, size of the central obscuration, and the footprint of the beam for each 
channel.
(c) Clear aperture of Channel 1's diffraction grating."""
    ))
    result.append(kgpy.latex.Label('fig:schematic'))
    return result
