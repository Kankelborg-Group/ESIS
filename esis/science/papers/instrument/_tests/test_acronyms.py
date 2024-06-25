import aastex
import esis.science.papers.instrument


def test_acronyms():
    result = esis.science.papers.instrument.acronyms()
    assert isinstance(result, list)
    for a in result:
        assert isinstance(a, aastex.Acronym)
