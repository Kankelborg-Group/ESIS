import dataclasses
import astropy.units as u
import esis


def test_timeline():
    result = esis.flights.f1.nsroc.timeline()

    assert isinstance(result, esis.nsroc.Timeline)

    for f in dataclasses.fields(result):
        assert getattr(result, f.name) > 0 * u.s
