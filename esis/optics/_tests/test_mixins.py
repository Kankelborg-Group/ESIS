import astropy.units as u
import named_arrays as na
from optika._tests import test_mixins
import esis


class AbstractTestCylindricallyTransformable(
    test_mixins.AbstractTestTransformable,
):
    def test_distance_radial(self, a: esis.optics.mixins.CylindricallyTransformable):
        result = a.distance_radial
        assert na.unit_normalized(result).is_equivalent(u.mm)

    def test_azimuth(self, a: esis.optics.mixins.CylindricallyTransformable):
        result = a.azimuth
        assert na.unit_normalized(result).is_equivalent(u.deg)
