import optika
import esis


def test_multilayer_AlSc():
    result = esis.flights.flight_02.optics.materials.multilayer_AlSc()
    assert isinstance(result, optika.materials.MultilayerMirror)
