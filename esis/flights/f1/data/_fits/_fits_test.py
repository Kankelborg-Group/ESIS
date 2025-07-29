import pathlib
import named_arrays as na
import esis


def test_path_fits():
    axis_time = "time"
    axis_chan = "channel"
    result = esis.flights.f1.data.path_fits(axis_time, axis_chan)
    assert isinstance(result, na.ScalarArray)
    assert result.shape[axis_chan] == 4
    for item in result.ndarray.flat:
        assert isinstance(item, pathlib.Path)
