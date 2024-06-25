import optika
import esis


def test_multilayer_AlSc():
    result = esis.flights.f2.optics.materials.multilayer_AlSc()
    assert isinstance(result, optika.materials.MultilayerMirror)


def test_multilayer_SiSc():
    result = esis.flights.f2.optics.materials.multilayer_SiSc()
    assert isinstance(result, optika.materials.MultilayerMirror)
