from . import cnn_inversion, level_1, level_3
import matplotlib.pyplot as plt
import numpy as np
import pytest


def test_cnn_inversion(capsys):
    with capsys.disabled():
        cnn_inversion()

@pytest.mark.skip('Jake\'s problem')
def test_level_1(capsys):
    with capsys.disabled():
        l1 = level_1()
        print(l1.intensity.shape)

# @pytest.mark.skip('Jake\'s problem')
def test_level_3(capsys):
    with capsys.disabled():
        l3 = level_3()