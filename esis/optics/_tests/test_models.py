import pytest
import astropy.units as u
import named_arrays as na
import optika._tests.test_mixins
import esis


class AbstractTestAbstractOpticsModel(
    optika._tests.test_mixins.AbstractTestPrintable,
):
    def test_name(self, a: esis.optics.abc.AbstractOpticsModel):
        result = a.name
        assert isinstance(result, str)

    def test_front_aperture(self, a: esis.optics.abc.AbstractOpticsModel):
        result = a.front_aperture
        if result is not None:
            assert isinstance(result, esis.optics.FrontAperture)

    def test_central_obscuration(self, a: esis.optics.abc.AbstractOpticsModel):
        result = a.central_obscuration
        if result is not None:
            assert isinstance(result, esis.optics.CentralObscuration)

    def test_primary_mirror(self, a: esis.optics.abc.AbstractOpticsModel):
        result = a.primary_mirror
        if result is not None:
            assert isinstance(result, esis.optics.PrimaryMirror)

    def test_field_stop(self, a: esis.optics.abc.AbstractOpticsModel):
        result = a.field_stop
        if result is not None:
            assert isinstance(result, esis.optics.FieldStop)

    def test_grating(self, a: esis.optics.abc.AbstractOpticsModel):
        result = a.grating
        if result is not None:
            assert isinstance(result, esis.optics.Grating)

    def test_filter(self, a: esis.optics.abc.AbstractOpticsModel):
        result = a.filter
        if result is not None:
            assert isinstance(result, esis.optics.Filter)

    def test_detector(self, a: esis.optics.abc.AbstractOpticsModel):
        result = a.detector
        if result is not None:
            assert isinstance(result, esis.optics.Detector)

    def test_grid_input_normalized(self, a: esis.optics.abc.AbstractOpticsModel):
        result = a.grid_input_normalized
        if result is not None:
            assert isinstance(result, optika.vectors.ObjectVectorArray)

    def test_angle_grating_input(self, a: esis.optics.abc.AbstractOpticsModel):
        result = a.angle_grating_input
        assert isinstance(na.as_named_array(result), na.AbstractArray)
        assert na.unit_normalized(result).is_equivalent(u.deg)

    def test_angle_grating_output(self, a: esis.optics.abc.AbstractOpticsModel):
        result = a.angle_grating_output
        assert isinstance(na.as_named_array(result), na.AbstractArray)
        assert na.unit_normalized(result).is_equivalent(u.deg)

    def test_wavelength_min(self, a: esis.optics.abc.AbstractOpticsModel):
        result = a.wavelength_min
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.AA)

    def test_wavelength_max(self, a: esis.optics.abc.AbstractOpticsModel):
        result = a.wavelength_max
        assert isinstance(na.as_named_array(result), na.AbstractScalar)
        assert na.unit_normalized(result).is_equivalent(u.AA)

    def test_system(self, a: esis.optics.abc.AbstractOpticsModel):
        result = a.system
        assert isinstance(result, optika.systems.AbstractSequentialSystem)
        assert result.surfaces


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.optics.OpticsModel(
            name="esis-test",
            front_aperture=esis.optics.FrontAperture(
                radius_clear=20 * u.imperial.inch,
            ),
            central_obscuration=esis.optics.CentralObscuration(
                num_folds=8,
            ),
            primary_mirror=esis.optics.PrimaryMirror(
                sag=optika.sags.ParabolicSag(-1000 * u.mm),
                num_folds=8,
                width_clear=100 * u.mm,
                width_border=1 * u.mm,
                material=optika.materials.Mirror(),
                translation=na.Cartesian3dVectorArray(z=2000) * u.mm,
            ),
            field_stop=esis.optics.FieldStop(
                num_folds=8,
                radius_clear=2 * u.mm,
                radius_mechanical=20 * u.mm,
                translation=na.Cartesian3dVectorArray(z=1000) * u.mm,
            ),
            grating=esis.optics.Grating(
                name="grating",
                serial_number="abc123",
                manufacturing_number="123abc",
                sag=optika.sags.SphericalSag(radius=500 * u.mm),
                material=optika.materials.Mirror(),
                rulings=optika.rulings.ConstantDensityRulings(5000 / u.mm),
                num_folds=8,
                halfwidth_inner=15 * u.mm,
                halfwidth_outer=10 * u.mm,
                width_border=1 * u.mm,
                width_border_inner=1.5 * u.mm,
                clearance=1 * u.mm,
                distance_radial=50 * u.mm,
                translation=na.Cartesian3dVectorArray(z=750) * u.mm,
                yaw=-5 * u.deg,
            ),
            filter=esis.optics.Filter(
                radius_clear=20 * u.mm,
                width_border=1 * u.mm,
                distance_radial=75 * u.mm,
                translation=na.Cartesian3dVectorArray(z=1750) * u.mm,
            ),
            detector=esis.optics.Detector(
                width_pixels=15 * u.um,
                shape_pixels=na.Cartesian2dVectorArray(2048, 1024),
                distance_radial=85 * u.mm,
                translation=na.Cartesian3dVectorArray(z=2000) * u.mm,
            ),
            grid_input_normalized=optika.vectors.ObjectVectorArray(
                wavelength=na.linspace(0, 1, num=3, axis="wavelength"),
                field=na.Cartesian2dVectorArray(
                    x=na.linspace(0, 1, num=5, axis="field_x"),
                    y=na.linspace(0, 1, num=5, axis="field_y"),
                ),
                pupil=na.Cartesian2dVectorArray(
                    x=na.linspace(0, 1, num=5, axis="pupil_x"),
                    y=na.linspace(0, 1, num=5, axis="pupil_y"),
                ),
            ),
        )
    ],
)
class TestOpticsModel(
    AbstractTestAbstractOpticsModel,
):
    pass
