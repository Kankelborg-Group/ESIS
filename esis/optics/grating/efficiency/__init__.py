import typing as typ
import pathlib
import astropy.units as u
import astropy.time
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

manufacturing_number = witness.manufacturing_number_g17
date_measurement = astropy.time.Time('2018-01-21')
wavelength_nominal = (63 * u.nm).to(u.AA)


def vs_angle_at_0aoi() -> typ.Tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity]:
    df = pandas.read_table(file_vs_angle_at_0aoi)
    angle_input = 0 * u.deg
    angle_output = df.index.to_numpy() * u.deg
    wavelength = 63 * u.nm
    efficiency = df.to_numpy() * u.dimensionless_unscaled
    return angle_input, angle_output, wavelength, efficiency


def vs_angle_at_3aoi() -> typ.Tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity]:
    df = pandas.read_table(file_vs_angle_at_3aoi)
    angle_input = 3 * u.deg
    angle_output = df.index.to_numpy() * u.deg
    wavelength = 63 * u.nm
    efficiency = df.to_numpy() * u.dimensionless_unscaled
    return angle_input, angle_output, wavelength, efficiency


def vs_position_x() -> typ.Tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity]:
    df = pandas.read_table(file_vs_position_x, sep=' ')
    arr = df.to_numpy()
    position_x = arr[..., 0] * u.mm
    position_y = 0 * u.mm
    wavelength = 63 * u.nm
    efficiency = arr[..., 1] * u.dimensionless_unscaled
    return position_x, position_y, wavelength, efficiency


def vs_position_y() -> typ.Tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity]:
    df = pandas.read_table(file_vs_position_y)
    arr = df.to_numpy()
    position_x = 0 * u.mm
    position_y = arr[..., 0] * u.mm
    wavelength = 63 * u.nm
    efficiency = arr[..., 1] * u.dimensionless_unscaled
    return position_x, position_y, wavelength, efficiency


def vs_wavelength() -> typ.Tuple[u.Quantity, u.Quantity, u.Quantity]:
    df = pandas.read_table(file_vs_wavelength, sep=' ')
    arr = df.to_numpy()
    angle_input = 0 * u.deg
    wavelength = arr[..., 0] * u.nm
    efficiency = arr[..., 1] * u.dimensionless_unscaled
    return angle_input, wavelength, efficiency

