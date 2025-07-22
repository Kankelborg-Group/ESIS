from .. import flight
from . import Level_0
import esis
import matplotlib.pyplot as plt


class TestLevel_0:

    def test_from_directory(self):
        level_0 = Level_0.from_directory(flight.raw_img_dir, detector=flight.optics.as_measured().detector)
        assert len(level_0.intensity.shape) == 4
        assert level_0.intensity.sum() > 0
        assert level_0.intensity.sum() > 0

    def test_foo(self):
        level_0 = esis.flight.level_0(caching=True)
        fig_sig_and_alt, ax_sig_and_alt = plt.subplots(figsize=(9, 5), constrained_layout=True)
        level_0.plot_altitude_and_signal_vs_time(ax=ax_sig_and_alt)
