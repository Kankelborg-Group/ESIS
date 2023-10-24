import pytest
import esis


@pytest.mark.parametrize("num_distribution", [0, 11])
def test_design_proposed(num_distribution: int):
    result = esis.flights.flight_02.optics.models.design_proposed(
        num_distribution=num_distribution,
    )
    assert isinstance(result, esis.optics.abc.AbstractOpticsModel)
