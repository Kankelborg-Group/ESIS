from . import document


def test_document(capsys):
    with capsys.disabled():
        doc = document()
        doc.generate_pdf()
        doc.generate_tex()
