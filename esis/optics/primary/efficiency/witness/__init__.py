import typing as typ
import pathlib
import astropy.units as u
import astropy.time
from esis.optics.grating.efficiency.witness import _vs_wavelength, angle_input

__all__ = [
    'vs_wavelength_p1',
    'vs_wavelength_p2',
    'angle_input'
]

base_path = pathlib.Path(__file__).parent
file_vs_wavelength_p1 = base_path / 'Witness_P1.txt'
file_vs_wavelength_p2 = base_path / 'Witness_P2.txt'
file_vs_wavelength_recoat_1 = base_path / 'mul063931.abs'


date_measurement = astropy.time.Time('2018-05-10')


def vs_wavelength_p1() -> typ.Tuple[str, u.Quantity, u.Quantity, u.Quantity]:
    return ('witness 01', ) + _vs_wavelength(file_vs_wavelength_p1)


def vs_wavelength_p2() -> typ.Tuple[str, u.Quantity, u.Quantity, u.Quantity]:
    return ('witness 02', ) +_vs_wavelength(file_vs_wavelength_p2)


def vs_wavelength_recoat_1() -> typ.Tuple[str, u.Quantity, u.Quantity, u.Quantity]:
    return ('recoat 01', ) + _vs_wavelength(file_vs_wavelength_recoat_1)
