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


def layout() -> matplotlib.figure.Figure:

    fig_layout, ax_layout = plt.subplots(figsize=(7.1, 4), constrained_layout=True)
    ax_layout.set_axis_off()
    # ax_layout.set_xticks([])
    # ax_layout.set_yticks([])
    # ax_layout.set_xlabel('')
    # ax_layout.set_ylabel('')
    ax_layout.set_aspect('equal')
    esis_optics = esis.optics.design.final(pupil_samples=kgpy.vector.Vector2D(3, 1), field_samples=1)
    esis_optics.pointing.y = 60 * u.deg
    esis_optics.central_obscuration = None
    esis_optics.filter = None

    esis_optics_rays = esis_optics.copy()
    chan_index = 4
    esis_optics_rays.grating.cylindrical_azimuth = esis_optics_rays.grating.cylindrical_azimuth[chan_index]
    # esis_optics_rays.filter.cylindrical_azimuth = esis_optics_rays.filter.cylindrical_azimuth[chan_index]
    esis_optics_rays.detector.cylindrical_azimuth = esis_optics_rays.detector.cylindrical_azimuth[chan_index]
    # esis_optics_rays.wavelength = esis_optics_rays.wavelength[0]

    esis_optics.system.plot(
        ax=ax_layout,
        components=('z', 'x'),
        plot_rays=False,
        plot_annotations=False,
    )
    _, colorbar = esis_optics_rays.system.plot(
        ax=ax_layout,
        components=('z', 'x'),
        # plot_rays=False,
        plot_annotations=False,
        # plot_vignetted=True,
    )
    colorbar.remove()

    return fig_layout


def layout_pdf() -> pathlib.Path:
    fig = layout()
    path = pathlib.Path(__file__).parent / 'layout_mpl.pdf'
    fig.savefig(path)
    plt.close(fig)
    return path
