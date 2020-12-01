import numpy as np
import matplotlib.pyplot as plt
import kgpy.optics
from . import design


def test_final(capsys):
    with capsys.disabled():
        esis = design.final(pupil_samples=5, field_samples=5)
        # esis.system.print_surfaces()
        # esis.system.plot_surfaces(list(esis.system), plot_rays=False, components=(0, 1))
        # plt.show()

        assert isinstance(esis.system, kgpy.optics.System)
        assert isinstance(esis.system.psf()[0], np.ndarray)


def test_final_from_poletto():
    esis = design.final_from_poletto(
        pupil_samples=5,
        field_samples=5,
        use_vls_grating=True,
    )
    assert isinstance(esis.system, kgpy.optics.System)
    assert isinstance(esis.system.psf()[0], np.ndarray)
