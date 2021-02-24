import pytest
from . import Level_4
import pathlib
import matplotlib.pyplot as plt


# @pytest.mark.skip('Jake\'s problem')
def test_from_pickle():
    path = pathlib.Path(__file__).parents[1] / 'flight/lev4_mart.pickle'
    lev4 = Level_4.from_pickle(path = path)

    # lev4.cube_list = lev4.best_inverted_results
    hyper_cube = lev4.plot()
    plt.show()