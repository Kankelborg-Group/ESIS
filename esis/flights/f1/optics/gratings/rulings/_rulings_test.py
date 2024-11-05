import optika
import esis


def test_ruling_design():
    r = esis.flights.f1.optics.gratings.rulings.ruling_design()
    assert isinstance(r, optika.rulings.AbstractRulings)


def test_ruling_measurement():
    r = esis.flights.f1.optics.gratings.rulings.ruling_measurement()
    assert isinstance(r, optika.rulings.AbstractRulings)
