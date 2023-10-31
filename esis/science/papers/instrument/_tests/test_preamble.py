import esis.science.papers.instrument


def test_preamble():
    result = esis.science.papers.instrument.preamble()
    assert isinstance(result, list)
