import numpy as np
from typing_extensions import Self
import dataclasses
import pathlib
import numpy.typing as npt
import named_arrays as na
import msfc_ccd
import esis

__all__ = [
    "Level_0",
]


@dataclasses.dataclass(eq=False, repr=False)
class Level_0(
    msfc_ccd.SensorData,
):
    """
    Representation of ESIS Level-0 images, the raw images gathered by the
    Data Acquisition and Control System (DACS).
    """

    timeline: None | esis.nsroc.Timeline = None
    """
    The sequence of NSROC events associated with these images.
    """

    @classmethod
    def from_fits(
        cls,
        path: str | pathlib.Path | na.AbstractScalarArray,
        sensor: msfc_ccd.abc.AbstractSensor,
        axis_x: str = "detector_x",
        axis_y: str = "detector_y",
        timeline: None | esis.nsroc.Timeline = None,
    ) -> Self:

        self = super().from_fits(
            path=path,
            sensor=sensor,
            axis_x=axis_x,
            axis_y=axis_y,
        )

        self.timeline = timeline

        self.inputs

        return self

    @property
    def channel(self) -> na.ScalarArray[npt.NDArray[str]]:
        """
        The name of each ESIS channel in a human-readable format.
        """

        sn = self.inputs.serial_number
        where_1 = sn == "6"
        where_2 = sn == "7"
        where_3 = sn == "9"
        where_4 = sn == "1"

        result = np.empty_like(sn, dtype=object)

        result[where_1] = "Channel 1"
        result[where_2] = "Channel 2"
        result[where_3] = "Channel 3"
        result[where_4] = "Channel 4"

        return result
