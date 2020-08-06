import astropy.units as u
from . import Grating


class TestGrating:

    def test_from_gregorian_layout(self):

        g = Grating.from_gregorian_layout(
            magnification=4,
            primary_focal_length=1 * u.m,
            primary_clear_radius=100 * u.mm,
            detector_channel_radius=120 * u.mm,
            detector_piston=-150 * u.mm,
            grating_mechanical_margin= 8 * u.mm,
        )

        assert isinstance(g, Grating)

        assert g.piston > 0
        assert g.channel_radius > 0
        assert g.outer_clear_radius > g.inner_clear_radius > 0
