import numpy as np
import named_arrays as na
import optika
import esis


def test_design():
    r = esis.flights.flight_01.optics.primary_mirrors.materials.multilayer_design()
    assert isinstance(r, optika.materials.AbstractMultilayerMirror)


def test_reflectivity_witness():
    r = esis.flights.flight_01.optics.primary_mirrors.materials.reflectivity_witness()
    assert isinstance(r, na.FunctionArray)
    assert isinstance(r.inputs, na.SpectralDirectionalVectorArray)
    assert isinstance(r.outputs, na.AbstractScalar)
    assert np.all(r.outputs >=0)


def test_multilayer_witness():
    r = esis.flights.flight_01.optics.primary_mirrors.materials.multilayer_witness()
    assert isinstance(r, optika.materials.MultilayerMirror)
