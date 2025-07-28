import pathlib
import numpy as np
import numpy.typing as npt
import named_arrays as na

__all__ = [
    "path_fits",
]


def path_fits(
    axis_time: str,
    axis_channel: str,
) -> na.ScalarArray[npt.NDArray[pathlib.Path]]:
    """
    Construct an array of paths to all the FITS files captured during
    the flight.

    Parameters
    ----------
    axis_time
        The name of the logical axis representing time.
    axis_channel
        The name of the logical axis representing the different channels.
    """
    path = pathlib.Path(__file__).parent.glob("*.fit.gz")
    path = sorted(list(path))
    path = np.array(path)
    path = na.ScalarArray(
        ndarray=path.reshape(4, -1),
        axes=(axis_channel, axis_time),
    )
    return path
