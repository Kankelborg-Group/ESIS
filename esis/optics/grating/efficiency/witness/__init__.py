import typing as typ
import pathlib
import astropy.units as u
import pandas

__all__ = [
    'vs_wavelength_g17',
    'vs_wavelength_g19',
    'vs_wavelength_g24'
]

base_path = pathlib.Path(__file__).parent
file_vs_wavelength_g17 = base_path / 'Witness_g17.txt'
file_vs_wavelength_g19 = base_path / 'Witness_g19.txt'
file_vs_wavelength_g24 = base_path / 'Witness_g24.txt'


def _vs_wavelength(file: pathlib.Path) -> typ.Tuple[u.Quantity, u.Quantity, u.Quantity]:
    df = pandas.read_table(file, sep=' ')
    arr = df.to_numpy()
    angle_input = 4 * u.deg
    wavelength = arr[..., 0] * u.nm
    efficiency = arr[..., 1] * u.dimensionless_unscaled
    return angle_input, wavelength, efficiency


def vs_wavelength_g17() -> typ.Tuple[str, u.Quantity, u.Quantity, u.Quantity]:
    return ('witness 017', ) + _vs_wavelength(file_vs_wavelength_g17)


def vs_wavelength_g19() -> typ.Tuple[str, u.Quantity, u.Quantity, u.Quantity]:
    return ('witness 019', ) +_vs_wavelength(file_vs_wavelength_g19)


def vs_wavelength_g24() -> typ.Tuple[str, u.Quantity, u.Quantity, u.Quantity]:
    return ('witness 024', ) +_vs_wavelength(file_vs_wavelength_g24)

