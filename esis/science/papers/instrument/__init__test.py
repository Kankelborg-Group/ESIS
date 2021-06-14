from . import document


def test_document():
    doc = document()
    doc.generate_pdf()
    doc.generate_tex()
