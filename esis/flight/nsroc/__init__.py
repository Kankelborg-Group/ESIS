import pathlib
from kgpy import Trajectory

__all__ = ['trajectory_file']

trajectory_file = pathlib.Path(__file__).parent / '36320_Trajectory.txt'


def trajectory() -> Trajectory:
    return Trajectory.from_nsroc_csv(csv_file=trajectory_file)

