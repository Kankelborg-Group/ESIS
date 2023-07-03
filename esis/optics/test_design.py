import numpy as np
import matplotlib.pyplot as plt
import kgpy.optics
from . import design


def test_final(capsys):
    with capsys.disabled():
        esis = design.final(
            pupil_samples=5,
            field_samples=5,
            # all_channels=False,
        )
        # esis.system.print_surfaces()
        # esis.system.plot_surfaces(list(esis.system), plot_rays=False, components=(0, 1))
        # plt.show()

        assert isinstance(esis.system, kgpy.optics.systems.System)

        print(esis.system.object_grid_normalized)

        fig, ax = plt.subplots(
            subplot_kw=dict(projection='3d'),
        )
        esis.system.plot(
            ax=ax,
            plot_annotations=False,
            # component_y='x',
        )
        ax.set_box_aspect((1600, 200, 200))
        ax.set_xlim(100, -1500)
        ax.set_ylim(-100,100)
        ax.set_zlim(-100,100)

        plt.show()

        # assert isinstance(esis.system.psf()[0], np.ndarray)


def test_final_from_poletto():
    esis = design.final_from_poletto(
        pupil_samples=5,
        field_samples=5,
        use_vls_grating=True,
    )
    assert isinstance(esis.system, kgpy.optics.System)
    assert isinstance(esis.system.psf()[0], np.ndarray)
