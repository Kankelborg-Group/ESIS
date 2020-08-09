import numpy as np
import kgpy.optics
from . import design


def test_final():
    esis = design.final(pupil_samples=5, field_samples=5)
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
