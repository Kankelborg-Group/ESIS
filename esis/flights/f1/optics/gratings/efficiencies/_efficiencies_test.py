import astropy.units as u
import numpy as np
import esis


def test_efficiency_vs_wavelength():
    result = esis.flights.f1.optics.gratings.efficiencies.efficiency_vs_wavelength()
    assert np.all(result.inputs.wavelength > 0 * u.nm)
    assert np.all(result.outputs > 0)


def test_efficiency_vs_x():
    result = esis.flights.f1.optics.gratings.efficiencies.efficiency_vs_x()
    assert np.all(result.inputs.position.unit.is_equivalent(u.mm))
    assert np.all(result.outputs > -0.01)


def test_efficiency_vs_y():
    result = esis.flights.f1.optics.gratings.efficiencies.efficiency_vs_y()
    assert np.all(result.inputs.position.unit.is_equivalent(u.mm))
    assert np.all(result.outputs > -0.01)


def test_efficiency_vs_angle_0deg():
    result = esis.flights.f1.optics.gratings.efficiencies.efficiency_vs_angle_0deg()
    assert np.all(result.inputs.direction.input.unit.is_equivalent(u.deg))
    assert np.all(result.outputs > -0.01)


def test_efficiency_vs_angle_3deg():
    result = esis.flights.f1.optics.gratings.efficiencies.efficiency_vs_angle_3deg()
    assert np.all(result.inputs.direction.input.unit.is_equivalent(u.deg))
    assert np.all(result.outputs > -0.01)
