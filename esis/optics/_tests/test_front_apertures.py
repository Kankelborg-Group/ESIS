import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika._tests.test_mixins
import esis


class AbstactTestAbstractFrontAperture(
    optika._tests.test_mixins.AbstractTestTranslatable,
):
    def test_radius_clear(
        self,
        a: esis.optics.abc.AbstractFrontAperture,
    ):
        result = a.radius_clear
        assert na.unit_normalized(result).is_equivalent(u.mm)
        assert np.all(result >= 0)

    def test_surface(self, a: esis.optics.abc.AbstractFrontAperture):
        result = a.surface
        assert isinstance(result, optika.surfaces.AbstractSurface)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.optics.FrontAperture(),
        esis.optics.FrontAperture(20 * u.imperial.inch),
    ],
)
class TestFrontAperture(
    AbstactTestAbstractFrontAperture,
):
    pass
