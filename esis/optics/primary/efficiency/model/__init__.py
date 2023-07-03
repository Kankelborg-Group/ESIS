import typing as typ
import pathlib
import astropy.units as u
import pandas
import kgpy.labeled
import kgpy.optics.vectors
import kgpy.function

__all__ = [
    'vs_wavelength',
]

base_path = pathlib.Path(__file__).parent

file_vs_wavelength = base_path / 'SiC_2nmSiO2.dat'


def vs_wavelength() -> kgpy.function.Array[kgpy.optics.vectors.InputAngleVector, kgpy.labeled.Array]:
    df = pandas.read_table(file_vs_wavelength, sep='\t', header=2)
    arr = df.to_numpy()
    angle_input = kgpy.labeled.Array([0] * u.deg, axes=['angle_input'])
    wavelength = kgpy.labeled.Array(arr[..., 0] * u.nm, axes=['wavelength'])
    efficiency = kgpy.labeled.Array(arr[..., 1] * u.dimensionless_unscaled, axes=['wavelength'])
    return kgpy.function.Array(
        input=kgpy.optics.vectors.InputAngleVector(
            wavelength=wavelength,
            angle_input_y=angle_input,
            # angle_input=kgpy.optics.vectors.Spherical(y=angle_input),
        ),
        output=efficiency,
    )

