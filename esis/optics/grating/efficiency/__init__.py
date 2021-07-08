import typing as typ
import pathlib
import astropy.units as u
import pandas
from . import witness

__all__ = [
    'vs_angle_at_0aoi',
    'vs_angle_at_3aoi',
    'vs_position_x',
    'vs_position_y',
    'vs_wavelength',
]

base_path = pathlib.Path(__file__).parent

file_vs_angle_at_0aoi = base_path / 'mul063284.txt'
file_vs_angle_at_3aoi = base_path / 'mul063281.txt'
file_vs_position_x = base_path / 'mul063283.abs'
file_vs_position_y = base_path / 'mul063282.txt'
file_vs_wavelength = base_path / 'mul063315.abs'


def vs_angle_at_0aoi() -> typ.Tuple[u.Quantity, u.Quantity]:
    df = pandas.read_table(file_vs_angle_at_0aoi)
    angle_output = df.index.to_numpy() * u.deg
    efficiency = 100 * df.to_numpy() * u.percent
    return angle_output, efficiency


def vs_angle_at_3aoi() -> typ.Tuple[u.Quantity, u.Quantity]:
    df = pandas.read_table(file_vs_angle_at_3aoi)
    angle_output = df.index.to_numpy() * u.deg
    efficiency = 100 * df.to_numpy() * u.percent
    return angle_output, efficiency


def vs_position_x() -> typ.Tuple[u.Quantity, u.Quantity]:
    df = pandas.read_table(file_vs_position_x, sep=' ')
    arr = df.to_numpy()
    position_x = arr[..., 0] * u.mm
    efficiency = 100 * arr[..., 1] * u.percent
    return position_x, efficiency


def vs_position_y() -> typ.Tuple[u.Quantity, u.Quantity]:
    df = pandas.read_table(file_vs_position_y)
    arr = df.to_numpy()
    position_y = arr[..., 0] * u.mm
    efficiency = 100 * arr[..., 1] * u.percent
    return position_y, efficiency


def vs_wavelength() -> typ.Tuple[u.Quantity, u.Quantity]:
    df = pandas.read_table(file_vs_wavelength, sep=' ')
    arr = df.to_numpy()
    wavelength = arr[..., 0] * u.nm
    efficiency = 100 * arr[..., 1] * u.percent
    return wavelength, efficiency

