import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika._tests.test_mixins
import esis


class AbstactTestAbstractFrontAperture(
    optika._tests.test_mixins.AbstractTestTranslatable,
):
    def test_surface(self, a: esis.optics.abc.AbstractFrontAperture):
        result = a.surface
        assert isinstance(result, optika.surfaces.AbstractSurface)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.optics.FrontAperture(),
    ],
)
class TestFrontAperture(
    AbstactTestAbstractFrontAperture,
):
    pass
