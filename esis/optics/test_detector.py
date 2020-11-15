import pytest
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from kgpy import vector, transform, optics
from . import Detector

__all__ = []


@pytest.fixture
def test_detector() -> Detector:
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

    def test_num_pixels_all(self, test_detector: Detector):
        assert test_detector.num_pixels_all[vector.ix] > test_detector.num_pixels[vector.ix]
        assert test_detector.num_pixels_all[vector.iy] == test_detector.num_pixels[vector.iy]

    def test_quadrants(self, test_detector: Detector):
        assert len(test_detector.quadrants) == 4
        for quad in test_detector.quadrants:
            img = np.zeros(test_detector.num_pixels_all[::-1])
            img[quad] = 1
            assert img.mean() == 1/4


    def test_pixel_half_width(self, test_detector: Detector):
        assert test_detector.pixel_half_width > 0

    def test_clear_width(self, test_detector: Detector):
        assert test_detector.clear_width > 0

    def test_clear_height(self, test_detector: Detector):
        assert test_detector.clear_height > 0

    def test_clear_half_width(self, test_detector: Detector):
        assert test_detector.clear_half_width > 0

    def test_clear_half_height(self, test_detector: Detector):
        assert test_detector.clear_half_height > 0

    def test_transform(self, test_detector: Detector):
        assert isinstance(test_detector.transform, transform.rigid.Transform)

    def test_surface(self, test_detector: Detector):
        assert isinstance(test_detector.surface, optics.Surface)
        assert isinstance(test_detector.surface.aperture, optics.Aperture)
        assert isinstance(test_detector.surface.aperture_mechanical, optics.Aperture)

    def test_convert_adu_to_electrons(self, test_detector: Detector):
        data = np.random.random(test_detector.gain.shape[:1] + test_detector.num_pixels) * u.adu
        data = test_detector.convert_adu_to_electrons(data)
        assert data.sum() > 0

    def test_convert_electrons_to_photons(self, test_detector: Detector):
        data = np.random.random(test_detector.gain.shape[:1] + test_detector.num_pixels) * u.electron
        data = test_detector.convert_electrons_to_photons(data, 171 * u.AA)
        assert data.sum() > 0

    def test_convert_adu_to_photons(self, test_detector: Detector):
        data = np.random.random(test_detector.gain.shape[:1] + test_detector.num_pixels) * u.adu
        data = test_detector.convert_adu_to_photons(data, 171 * u.AA)
        assert data.sum() > 0

    def test_remove_inactive_pixels(self, test_detector: Detector):
        data = np.random.random(test_detector.num_pixels_all[::-1]) * u.adu
        data = test_detector.remove_inactive_pixels(data)
        assert data.shape == test_detector.num_pixels[::-1]
        assert data.sum() > 0

    def test_readout_noise_image(self, test_detector: Detector):
        img = test_detector.readout_noise_image
        assert np.isclose(img.mean(), test_detector.readout_noise.mean())
