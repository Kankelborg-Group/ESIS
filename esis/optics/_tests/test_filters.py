import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import optika._tests.test_mixins
import esis
from . import test_mixins


class AbstractTestAbstractFilter(
    optika._tests.test_mixins.AbstractTestPrintable,
    optika._tests.test_mixins.AbstractTestRollable,
    optika._tests.test_mixins.AbstractTestYawable,
    optika._tests.test_mixins.AbstractTestPitchable,
    optika._tests.test_mixins.AbstractTestTranslatable,
    test_mixins.AbstractTestCylindricallyTransformable,
):
    def test_material(
        self,
        a: esis.optics.abc.AbstractFilter,
    ):
        result = a.material
        if result is not None:
            assert isinstance(result, optika.materials.AbstractMaterial)

    def test_material_oxide(
        self,
        a: esis.optics.abc.AbstractFilter,
    ):
        result = a.material_oxide
        if result is not None:
            assert isinstance(result, optika.materials.AbstractMaterial)

    def test_material_mesh(
        self,
        a: esis.optics.abc.AbstractFilter,
    ):
        result = a.material_mesh
        if result is not None:
            assert isinstance(result, optika.materials.AbstractMaterial)

    def test_ratio_mesh(
        self,
        a: esis.optics.abc.AbstractFilter,
    ):
        result = a.ratio_mesh
        assert na.unit_normalized(result).is_equivalent(u.dimensionless_unscaled)
        assert np.all(result >= 0)

    def test_frequency_mesh(
        self,
        a: esis.optics.abc.AbstractFilter,
    ):
        result = a.frequency_mesh
        assert na.unit_normalized(result).is_equivalent(1 / u.mm)
        assert np.all(result >= 0)

    def test_radius_clear(
        self,
        a: esis.optics.abc.AbstractFilter,
    ):
        result = a.radius_clear
        assert na.unit_normalized(result).is_equivalent(u.mm)
        assert np.all(result >= 0)

    def test_width_border(
        self,
        a: esis.optics.abc.AbstractFilter,
    ):
        result = a.width_border
        assert na.unit_normalized(result).is_equivalent(u.mm)
        assert np.all(result >= 0)

    def test_thickness(
        self,
        a: esis.optics.abc.AbstractFilter,
    ):
        result = a.thickness
        assert na.unit_normalized(result).is_equivalent(u.mm)
        assert np.all(result >= 0)

    def test_thickness_oxide(
        self,
        a: esis.optics.abc.AbstractFilter,
    ):
        result = a.thickness_oxide
        assert na.unit_normalized(result).is_equivalent(u.mm)
        assert np.all(result >= 0)

    def test_surfaces(
        self,
        a: esis.optics.abc.AbstractFilter,
    ):
        result = a.surfaces
        assert isinstance(result, list)
        for surface in result:
            assert isinstance(surface, optika.surfaces.AbstractSurface)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.optics.Filter(),
        esis.optics.Filter(
            ratio_mesh=30 * u.percent,
            frequency_mesh=10 / u.mm,
            radius_clear=15 * u.mm,
            width_border=1 * u.mm,
            thickness=100 * u.nm,
            thickness_oxide=10 * u.nm,
            distance_radial=100 * u.mm,
            azimuth=45 * u.deg,
            yaw=15 * u.deg,
        ),
    ],
)
class TestFilter(
    AbstractTestAbstractFilter,
):
    pass
