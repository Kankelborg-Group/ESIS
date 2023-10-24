import pytest
import esis.optics._tests.test_models


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design_full(num_distribution: int):
    result = esis.flights.flight_01.optics.models.design_full(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractOpticsModel)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design(num_distribution: int):
    result = esis.flights.flight_01.optics.models.design(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractOpticsModel)


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design_single(num_distribution: int):
    result = esis.flights.flight_01.optics.models.design_single(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractOpticsModel)
