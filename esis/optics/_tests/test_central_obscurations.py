import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
from optika._tests import test_mixins
import esis


class AbstractTestAbstractCentralObscuration(
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestTranslatable,
):
    def test_num_folds(self, a: esis.optics.abc.AbstractCentralObscuration):
        result = a.num_folds
        assert isinstance(result, int)
        assert result >= 0

    def test_halfwidth(self, a: esis.optics.abc.AbstractCentralObscuration):
        result = a.halfwidth
        assert na.unit_normalized(result).is_equivalent(u.mm)
        assert np.all(result >= 0)

    def test_radiush(self, a: esis.optics.abc.AbstractCentralObscuration):
        result = a.radius
        assert na.unit_normalized(result).is_equivalent(u.mm)
        assert np.all(result >= 0)

    def test_remove_last_vertex(self, a: esis.optics.abc.AbstractCentralObscuration):
        result = a.remove_last_vertex
        assert isinstance(result, bool)

    def test_surface(self, a: esis.optics.abc.AbstractCentralObscuration):
        result = a.surface
        assert isinstance(result, optika.surfaces.AbstractSurface)
        assert isinstance(result.aperture, optika.apertures.AbstractAperture)
        assert np.all(result.aperture.inverted)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.optics.CentralObscuration(
            num_folds=8,
            halfwidth=50 * u.mm,
            remove_last_vertex=True,
            translation=na.Cartesian3dVectorArray(z=100) * u.mm,
        )
    ],
)
class TestCentralObscuration(
    AbstractTestAbstractCentralObscuration,
):
    pass
