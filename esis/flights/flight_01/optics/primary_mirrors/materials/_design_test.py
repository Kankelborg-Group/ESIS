import optika
import esis


def test_design():
    result = esis.flights.flight_01.optics.primary_mirrors.materials.design()
    assert isinstance(result, optika.materials.AbstractMultilayerMirror)
