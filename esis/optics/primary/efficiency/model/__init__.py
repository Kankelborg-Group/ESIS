import typing as typ
import pathlib
import astropy.units as u
import pandas

__all__ = [
    'vs_wavelength',
]

base_path = pathlib.Path(__file__).parent

file_vs_wavelength = base_path / 'SiC_2nmSiO2.dat'


def vs_wavelength() -> typ.Tuple[u.Quantity, u.Quantity, u.Quantity]:
    df = pandas.read_table(file_vs_wavelength, sep='\t', header=2)
    arr = df.to_numpy()
    angle_input = 0 * u.deg
    wavelength = arr[..., 0] * u.nm
    efficiency = arr[..., 1] * u.dimensionless_unscaled
    return angle_input, wavelength, efficiency

