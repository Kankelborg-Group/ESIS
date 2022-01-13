import pathlib
import matplotlib.figure
import matplotlib.colors
import astropy.units as u
import kgpy.vector
import esis.optics
from . import formatting
from . import caching

__all__ = [
    'pupil_samples',
    'field_samples',
]


pupil_samples = 201
field_samples = 4


def figure_mpl() -> matplotlib.figure.Figure:
    optics = esis.optics.design.final(
        pupil_samples=pupil_samples,
        pupil_is_stratified_random=True,
        field_samples=field_samples,
        all_channels=False,
    )
    optics.num_emission_lines = 1

    rays = optics.rays_output

    bins = rays.input_grid.pupil.num_samples_normalized.x // 2

    limit_max = 0.5 * u.pix * kgpy.vector.Vector2D()
    fig, axs = rays.plot_pupil_hist2d_vs_field(
        wavlen_index=0,
        norm=matplotlib.colors.PowerNorm(1 / 3),
        bins=bins,
        cmap='gray_r',
        limit_min=-limit_max,
        limit_max=limit_max,
    )
    fig.set_figheight(2.6)
    fig.set_figwidth(formatting.column_width)
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


def pdf() -> pathlib.Path:
    return caching.cache_pdf(figure_mpl)
