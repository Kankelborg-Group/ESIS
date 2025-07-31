import esis

def test_level_0():

    lvl0 = esis.flights.f1.data.level_0()

    assert isinstance(lvl0, esis.data.Level_0)
