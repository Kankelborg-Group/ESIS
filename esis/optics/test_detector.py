import pytest
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from kgpy import vector, transform, optics
from . import Detector

__all__ = []


@pytest.fixture
def detector_example() -> Detector:
    return Detector(
        inclination=5 * u.deg,
        roll=.2 * u.deg,
        twist=-.1 * u.deg,
        pixel_width=10 * u.um,
        num_pixels=(1024, 2048),
        border_width_right=1*u.mm,
        border_width_left=1 * u.mm,
        border_width_top=1 * u.mm,
        border_width_bottom=1 * u.mm,
        dynamic_clearance=10 * u.mm,
        npix_blank=5,
        npix_overscan=50,
        gain=[
            [2.3, 2.4, 2.5, 2.6],
            [2.1, 2.2, 2.3, 2.4],
            [2.5, 2.4, 2.3, 2.1],
            [2.0, 2.2, 2.6, 2.3],
        ] * u.electron / u.adu,
        readout_noise=[
            [4.1, 4.0, 4.0, 4.0],
            [4.0, 4.0, 4.2, 4.0],
            [4.0, 4.4, 4.0, 4.0],
            [4.0, 4.0, 4.3, 3.9],
        ] * u.adu,
    )


class TestDetector:

    def test_num_pixels_all(self, detector_example):
        assert detector_example.num_pixels_all[vector.ix] > detector_example.num_pixels[vector.ix]
        assert detector_example.num_pixels_all[vector.iy] == detector_example.num_pixels[vector.iy]

    def test_quadrants(self, detector_example):
        assert len(detector_example.quadrants) == 4
        for quad in detector_example.quadrants:
            img = np.zeros(detector_example.num_pixels_all[::-1])
            img[quad] = 1
            assert img.mean() == 1/4

    def test_pixel_half_width(self, detector_example):
        assert detector_example.pixel_half_width > 0

    def test_clear_width(self, detector_example):
        assert detector_example.clear_width > 0

    def test_clear_height(self, detector_example):
        assert detector_example.clear_height > 0

    def test_clear_half_width(self, detector_example):
        assert detector_example.clear_half_width > 0

    def test_clear_half_height(self, detector_example):
        assert detector_example.clear_half_height > 0

    def test_transform(self, detector_example):
        assert isinstance(detector_example.transform, transform.rigid.Transform)

    def test_surface(self, detector_example):
        assert isinstance(detector_example.surface, optics.surface.Surface)
        assert isinstance(detector_example.surface.aperture, optics.surface.aperture.Aperture)
        assert isinstance(detector_example.surface.aperture_mechanical, optics.surface.aperture.Aperture)

    def test_convert_adu_to_electrons(self, detector_example):
        data = np.random.random(detector_example.gain.shape[:1] + detector_example.num_pixels) * u.adu
        data = detector_example.convert_adu_to_electrons(data)
        assert data.sum() > 0

    def test_convert_electrons_to_photons(self, detector_example):
        data = np.random.random(detector_example.gain.shape[:1] + detector_example.num_pixels) * u.electron
        data = detector_example.convert_electrons_to_photons(data, 171 * u.AA)
        assert data.sum() > 0

    def test_convert_adu_to_photons(self, detector_example):
        data = np.random.random(detector_example.gain.shape[:1] + detector_example.num_pixels) * u.adu
        data = detector_example.convert_adu_to_photons(data, 171 * u.AA)
        assert data.sum() > 0

    def test_remove_inactive_pixels(self, detector_example):
        data = np.random.random(detector_example.num_pixels_all[::-1]) * u.adu
        data = detector_example.remove_inactive_pixels(data)
        assert data.shape == detector_example.num_pixels[::-1]
        assert data.sum() > 0

    def test_readout_noise_image(self, detector_example):
        img = detector_example.readout_noise_image(4)
        assert np.isclose(img.mean(), detector_example.readout_noise.mean())
