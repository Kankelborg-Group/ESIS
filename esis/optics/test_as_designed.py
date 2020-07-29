import kgpy.optics
from . import design


def test_esis_as_designed():
    esis = design(pupil_samples=5, field_samples=5)
    assert isinstance(esis.system, kgpy.optics.System)
    assert isinstance(esis.system.image_rays, kgpy.optics.Rays)
