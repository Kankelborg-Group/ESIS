import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import optika._tests.test_mixins
import esis


class AbstractTestAbstractFieldStop(
    optika._tests.test_mixins.AbstractTestPrintable,
    optika._tests.test_mixins.AbstractTestTranslatable,
):
    def test_num_folds(
        self,
        a: esis.optics.abc.AbstractFieldStop,
    ):
        result = a.num_folds
        assert isinstance(result, int)
        assert result >= 0

    def test_num_sides(
        self,
        a: esis.optics.abc.AbstractFieldStop,
    ):
        result = a.num_sides
        assert isinstance(result, int)
        assert result >= 0

    def test_radius_clear(
        self,
        a: esis.optics.abc.AbstractFieldStop,
    ):
        result = a.radius_clear
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.mm)
        assert np.all(result >= 0)

    def test_width_clear(
        self,
        a: esis.optics.abc.AbstractFieldStop,
    ):
        result = a.width_clear
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.mm)

    def test_radius_mechanical(
        self,
        a: esis.optics.abc.AbstractFieldStop,
    ):
        result = a.radius_mechanical
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.mm)
        assert np.all(result >= 0)

    def test_surface(
        self,
        a: esis.optics.abc.AbstractFieldStop,
    ):
        result = a.surface
        assert isinstance(result, optika.surfaces.Surface)
        assert result.is_field_stop
        assert result.aperture is not None
        assert result.aperture_mechanical is not None


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.optics.FieldStop(),
        esis.optics.FieldStop(
            num_folds=8,
            radius_clear=5 * u.mm,
            radius_mechanical=10 * u.mm,
            translation=na.Cartesian3dVectorArray(z=200) * u.mm,
        ),
    ],
)
class TestFieldStop(
    AbstractTestAbstractFieldStop,
):
    pass
