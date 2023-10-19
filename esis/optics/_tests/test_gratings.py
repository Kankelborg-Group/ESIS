import pytest
import astropy.units as u
import named_arrays as na
import optika
from optika._tests import test_mixins
import esis


class AbstractTestAbstractGrating(
    test_mixins.AbstractTestPrintable,
    test_mixins.AbstractTestRollable,
    test_mixins.AbstractTestYawable,
    test_mixins.AbstractTestPitchable,
    test_mixins.AbstractTestTranslatable,
):
    def test_name(
        self,
        a: esis.optics.abc.AbstractGrating,
    ):
        assert isinstance(a.name, str)

    def test_serial_number(
        self,
        a: esis.optics.abc.AbstractGrating,
    ):
        assert isinstance(a.serial_number, str)

    def test_manufacturing_number(
        self,
        a: esis.optics.abc.AbstractGrating,
    ):
        assert isinstance(a.manufacturing_number, str)

    def test_angle_input(
        self,
        a: esis.optics.abc.AbstractGrating,
    ):
        result = a.angle_input
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.deg)

    def test_angle_output(
        self,
        a: esis.optics.abc.AbstractGrating,
    ):
        result = a.angle_output
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.deg)

    def test_sag(
        self,
        a: esis.optics.abc.AbstractGrating,
    ):
        assert isinstance(a.sag, optika.sags.AbstractSag)

    def test_material(
        self,
        a: esis.optics.abc.AbstractGrating,
    ):
        assert isinstance(a.material, optika.materials.AbstractMaterial)

    def test_rulings(
        self,
        a: esis.optics.abc.AbstractGrating,
    ):
        assert isinstance(a.rulings, optika.rulings.AbstractRulings)

    def test_num_folds(
        self,
        a: esis.optics.abc.AbstractGrating,
    ):
        assert isinstance(a.num_folds, int)
        assert a.num_folds >= 0

    def test_angle_aperture(
        self,
        a: esis.optics.abc.AbstractGrating,
    ):
        result = a.angle_aperture
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.deg)

    def test_halfwidth_inner(
        self,
        a: esis.optics.abc.AbstractGrating,
    ):
        result = a.halfwidth_inner
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.mm)

    def test_halfwidth_outer(
        self,
        a: esis.optics.abc.AbstractGrating,
    ):
        result = a.halfwidth_outer
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.mm)

    def test_width_border(
        self,
        a: esis.optics.abc.AbstractGrating,
    ):
        result = a.width_border
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.mm)

    def test_width_border_inner(
        self,
        a: esis.optics.abc.AbstractGrating,
    ):
        result = a.width_border_inner
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.mm)

    def test_clearance(
        self,
        a: esis.optics.abc.AbstractGrating,
    ):
        result = a.clearance
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.mm)

    def test_surface(
        self,
        a: esis.optics.abc.AbstractGrating,
    ):
        assert isinstance(a.surface, optika.surfaces.AbstractSurface)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.optics.Grating(
            name="gratings",
            serial_number="1234",
            manufacturing_number="ABCD",
            sag=optika.sags.SphericalSag(500 * u.mm),
            material=optika.materials.Mirror(),
            rulings=optika.rulings.ConstantDensityRulings(5000 / u.mm),
            num_folds=8,
            halfwidth_inner=10 * u.mm,
            halfwidth_outer=15 * u.mm,
            width_border=1 * u.mm,
            width_border_inner=2 * u.mm,
            clearance=1 * u.mm,
            distance_radial=50 * u.mm,
            yaw=20 * u.deg,
        )
    ],
)
class TestGrating(
    AbstractTestAbstractGrating,
):
    pass
