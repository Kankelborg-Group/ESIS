import aastex
import esis.science.papers.instrument


def test_authors():
    result = esis.science.papers.instrument.authors()
    assert isinstance(result, list)
    for author in result:
        assert isinstance(author, aastex.Author)
