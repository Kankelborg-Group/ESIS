import typing as typ
import pathlib
import astropy.units as u
import pandas
from esis.optics.grating.efficiency.witness import _vs_wavelength

__all__ = [
    'vs_wavelength_p1',
    'vs_wavelength_p2',
]

base_path = pathlib.Path(__file__).parent
file_vs_wavelength_p1 = base_path / 'Witness_P1.txt'
file_vs_wavelength_p2 = base_path / 'Witness_P2.txt'


def vs_wavelength_p1() -> typ.Tuple[str, u.Quantity, u.Quantity, u.Quantity]:
    return ('witness 01', ) + _vs_wavelength(file_vs_wavelength_p1)


def vs_wavelength_p2() -> typ.Tuple[str, u.Quantity, u.Quantity, u.Quantity]:
    return ('witness 02', ) +_vs_wavelength(file_vs_wavelength_p2)
