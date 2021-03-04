from . import cnn_inversion


def test_cnn_inversion(capsys):
    with capsys.disabled():
        cnn_inversion()