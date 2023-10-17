import pytest
import abc
import astropy.units as u
import named_arrays as na
import optika
import esis


class AbstractTestAbstractPrimaryMirror(
    abc.ABC,
):
    def test_name(
        self,
        a: esis.optics.abc.AbstractPrimaryMirror,
    ):
        assert isinstance(a.name, str)

    def test_sag(
        self,
        a: esis.optics.abc.AbstractPrimaryMirror,
    ):
        assert isinstance(a.sag, optika.sags.AbstractSag)

    def test_num_folds(
        self,
        a: esis.optics.abc.AbstractPrimaryMirror,
    ):
        assert isinstance(a.num_folds, int)
        assert a.num_folds > 2

    def test_width_clear(
        self,
        a: esis.optics.abc.AbstractPrimaryMirror,
    ):
        assert isinstance(na.as_named_array(a.width_clear), na.AbstractScalar)
        assert na.unit_normalized(a.width_clear).is_equivalent(u.mm)

    def test_radius_clear(
        self,
        a: esis.optics.abc.AbstractPrimaryMirror,
    ):
        assert isinstance(na.as_named_array(a.radius_clear), na.AbstractScalar)
        assert na.unit_normalized(a.radius_clear).is_equivalent(u.mm)

    def test_width_border(
        self,
        a: esis.optics.abc.AbstractPrimaryMirror,
    ):
        assert isinstance(na.as_named_array(a.width_border), na.AbstractScalar)
        assert na.unit_normalized(a.width_border).is_equivalent(u.mm)

    def test_radius_mechanical(
        self,
        a: esis.optics.abc.AbstractPrimaryMirror,
    ):
        assert isinstance(na.as_named_array(a.radius_mechanical), na.AbstractScalar)
        assert na.unit_normalized(a.radius_mechanical).is_equivalent(u.mm)

    def test_material(
        self,
        a: esis.optics.abc.AbstractPrimaryMirror,
    ):
        assert isinstance(a.material, optika.materials.AbstractMaterial)

    def test_surface(
        self,
        a: esis.optics.abc.AbstractPrimaryMirror,
    ):
        assert isinstance(a.surface, optika.surfaces.AbstractSurface)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.optics.PrimaryMirror(
            name="primary",
            sag=optika.sags.ParabolicSag(1000 * u.mm),
            num_folds=8,
            material=optika.materials.Mirror(),
        ),
    ],
)
class TestPrimaryMirror(
    AbstractTestAbstractPrimaryMirror,
):
    pass
