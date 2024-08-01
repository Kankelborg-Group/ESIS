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
            assert isinstance(result, optika.materials.AbstractThinFilmFilter)

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

    def test_surface(
        self,
        a: esis.optics.abc.AbstractFilter,
    ):
        result = a.surface
        assert isinstance(result, optika.surfaces.AbstractSurface)


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.optics.Filter(),
        esis.optics.Filter(
            material=optika.materials.ThinFilmFilter(
                layer=optika.materials.Layer(
                    chemical="Al",
                    thickness=100 * u.nm,
                ),
                layer_oxide=optika.materials.Layer(
                    chemical="Al2O3",
                    thickness=4 * u.nm,
                ),
                mesh=optika.materials.meshes.Mesh(
                    chemical="Ni",
                    efficiency=0.75,
                    pitch=1 * u.um,
                ),
            ),
            radius_clear=15 * u.mm,
            width_border=1 * u.mm,
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
