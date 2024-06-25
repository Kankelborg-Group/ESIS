import pytest
import astropy.units as u
import optika._tests.test_mixins
import esis


@pytest.mark.parametrize(
    argnames="a",
    argvalues=[
        esis.optics.Requirements(
            resolution_spatial=1.5 * u.Mm,
            resolution_spectral=18 * u.km / u.s,
            fov=10 * u.arcmin,
            snr=17.3 * u.dimensionless_unscaled,
            cadence=15 * u.s,
            length_observation=150 * u.s,
        )
    ],
)
class TestRequirements(
    optika._tests.test_mixins.AbstractTestPrintable,
):

    def test_resolution_spatial(self, a: esis.optics.Requirements):
        assert a.resolution_spatial > (0 * u.m)

    def test_resolution_spectral(self, a: esis.optics.Requirements):
        assert a.resolution_spectral > (0 * u.km / u.s)

    def test_fov(self, a: esis.optics.Requirements):
        assert a.fov > (0 * u.arcmin)

    def test_snr(self, a: esis.optics.Requirements):
        assert a.snr > 0

    def test_cadence(self, a: esis.optics.Requirements):
        assert a.cadence > (0 * u.s)

    def test_length_observation(self, a: esis.optics.Requirements):
        assert a.length_observation > (0 * u.s)

    def test_resolution_angular(self, a: esis.optics.Requirements):
        assert a.resolution_angular > (0 * u.arcsec)
