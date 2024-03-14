import optika
import esis


def test_multilayer_design():
    r = esis.flights.f1.optics.primaries.materials.multilayer_design()
    assert isinstance(r, optika.materials.AbstractMultilayerMirror)


def test_multilayer_witness_measured():
    r = esis.flights.f1.optics.primaries.materials.multilayer_witness_measured()
    assert isinstance(r, optika.materials.MeasuredMirror)


def test_multilayer_witness_fit():
    r = esis.flights.f1.optics.primaries.materials.multilayer_witness_fit()
    assert isinstance(r, optika.materials.MultilayerMirror)


def test_multilayer_fit():
    r = esis.flights.f1.optics.primaries.materials.multilayer_fit()
    assert isinstance(r, optika.materials.MultilayerMirror)
