import typing as typ
import pathlib
import astropy.units as u
import pandas
import kgpy.labeled
import kgpy.optics.vectors
import kgpy.function

__all__ = [
    'vs_wavelength_g17',
    'vs_wavelength_g19',
    'vs_wavelength_g24'
]

base_path = pathlib.Path(__file__).parent
file_vs_wavelength_g17 = base_path / 'Witness_g17.txt'
file_vs_wavelength_g19 = base_path / 'Witness_g19.txt'
file_vs_wavelength_g24 = base_path / 'Witness_g24.txt'

manufacturing_number_g17 = 'UBO-16-017'
manufacturing_number_g19 = 'UBO-16-019'
manufacturing_number_g24 = 'UBO-16-024'
manufacturing_number_unmeasured = 'UBO-16-014'

angle_input = kgpy.labeled.Array([4] * u.deg, axes=['angle_input'])


def _vs_wavelength(file: pathlib.Path) -> kgpy.function.Array[kgpy.optics.vectors.InputAngleVector, kgpy.labeled.Array]:
    df = pandas.read_table(file, sep=' ')
    arr = df.to_numpy()
    wavelength = kgpy.labeled.Array(arr[..., 0] * u.nm, axes=['wavelength'])
    efficiency = kgpy.labeled.Array(arr[..., 1] * u.dimensionless_unscaled, axes=['wavelength'])
    return kgpy.function.Array(
        input=kgpy.optics.vectors.InputAngleVector(
            wavelength=wavelength,
            angle_input_y=angle_input,
            # angle_input=angle_input,
        ),
        output=efficiency,
    )


def vs_wavelength_g17() -> typ.Tuple[str, kgpy.function.Array[kgpy.optics.vectors.InputAngleVector, kgpy.labeled.Array]]:
    return manufacturing_number_g17, _vs_wavelength(file_vs_wavelength_g17)


def vs_wavelength_g19() -> typ.Tuple[str, kgpy.function.Array[kgpy.optics.vectors.InputAngleVector, kgpy.labeled.Array]]:
    return manufacturing_number_g19, _vs_wavelength(file_vs_wavelength_g19)


def vs_wavelength_g24() -> typ.Tuple[str, kgpy.function.Array[kgpy.optics.vectors.InputAngleVector, kgpy.labeled.Array]]:
    return manufacturing_number_g24, _vs_wavelength(file_vs_wavelength_g24)

