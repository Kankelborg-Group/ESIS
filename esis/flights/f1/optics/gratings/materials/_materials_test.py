import optika
import esis


def test_multilayer_design():
    r = esis.flights.f1.optics.gratings.materials.multilayer_design()
    assert isinstance(r, optika.materials.AbstractMultilayerMirror)


def test_multilayer_witness_measured():
    r = esis.flights.f1.optics.gratings.materials.multilayer_witness_measured()
    assert isinstance(r, optika.materials.MeasuredMirror)
