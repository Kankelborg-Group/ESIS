import pytest
import matplotlib.pyplot as plt
import matplotlib.colors
import astropy.units as u
import kgpy.optics
from . import Optics


class TestOptics:

    def test_esis_tvls_from_poletto(self, capsys):
        with capsys.disabled():
            n = 7
            esis_sparse = Optics.esis_from_poletto(pupil_samples=5, field_samples=5)
            esis = Optics.esis_from_poletto(pupil_samples=20, field_samples=5)

            esis_sparse.system.plot_projections(
                # plot_vignetted=True,
                color_axis=kgpy.optics.Rays.axis.field_x,
            )
            esis.system.plot_footprint(
                surf=esis.components.field_stop.surface,
                # plot_vignetted=True,
                color_axis=kgpy.optics.Rays.axis.field_x,
            )
            esis.system.plot_footprint(
                surf=esis.components.grating.surface,
                # plot_vignetted=True,
                color_axis=kgpy.optics.Rays.axis.field_x,
            )
            esis.system.plot_footprint(
                # surf=esis.components.grating.surface,
                # plot_vignetted=True,
                color_axis=kgpy.optics.Rays.axis.pupil_x,
            )
            rays = esis.system.image_rays.copy()
            rays.position = (rays.position / (2 * esis.components.detector.pix_half_width_x / u.pix)).to(u.pix)
            for w in range(esis.wavelengths.shape[~0]):
                rays.plot_pupil_hist2d_vs_field(
                    wavlen_index=w,
                    relative_to_centroid=(True, True),
                    norm=matplotlib.colors.PowerNorm(1/2),
                )
            plt.show()

