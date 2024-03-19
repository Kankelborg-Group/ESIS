import optika
import esis


def test_thin_film_design():
    r = esis.flights.f1.optics.filters.materials.thin_film_design()
    assert isinstance(r, optika.materials.ThinFilmFilter)
