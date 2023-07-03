import typing as typ
import pathlib
import astropy.units as u
import astropy.time
import pandas
import kgpy.labeled
import kgpy.vectors
import kgpy.optics.vectors
import kgpy.function
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


def vs_angle_at_0aoi() -> kgpy.function.Array[kgpy.optics.vectors.InputOutputAngleVector, kgpy.labeled.Array]:
    df = pandas.read_table(file_vs_angle_at_0aoi)
    angle_input = kgpy.labeled.Array([0] * u.deg, axes=['angle_input'])
    angle_output = kgpy.labeled.Array(df.index.to_numpy() * u.deg, axes=['angle_output'])
    wavelength = kgpy.labeled.Array([63] * u.nm, axes=['wavelength'])
    efficiency = kgpy.labeled.Array(df.to_numpy() * u.dimensionless_unscaled, axes=['angle_output'])
    return kgpy.function.Array(
        input=kgpy.optics.vectors.InputOutputAngleVector(
            wavelength=wavelength,
            angle_input_y=angle_input,
            angle_output_y=angle_output,
            # angle_input=kgpy.optics.vectors.Spherical(y=angle_input),
            # angle_output=kgpy.optics.vectors.Spherical(y=angle_output),
        ),
        output=efficiency,
    )


def vs_angle_at_3aoi() -> kgpy.function.Array[kgpy.optics.vectors.InputOutputAngleVector, kgpy.labeled.Array]:
    df = pandas.read_table(file_vs_angle_at_3aoi)
    angle_input = kgpy.labeled.Array([3] * u.deg, axes=['angle_input'])
    angle_output = kgpy.labeled.Array(df.index.to_numpy() * u.deg, axes=['angle_output'])
    wavelength = kgpy.labeled.Array([63] * u.nm, axes=['wavelength'])
    efficiency = kgpy.labeled.Array(df.to_numpy() * u.dimensionless_unscaled, axes=['angle_output'])
    return kgpy.function.Array(
        input=kgpy.optics.vectors.InputOutputAngleVector(
            wavelength=wavelength,
            angle_input_y=angle_input,
            angle_output_y=angle_output,
            # angle_input=kgpy.optics.vectors.Spherical(y=angle_input),
            # angle_output=kgpy.optics.vectors.Spherical(y=angle_output),
        ),
        output=efficiency,
    )


def vs_position_x() -> kgpy.function.Array[kgpy.optics.vectors.SpectralPositionVector, kgpy.labeled.Array]:
    df = pandas.read_table(file_vs_position_x, sep=' ')
    arr = df.to_numpy()
    position_x = kgpy.labeled.Array(arr[..., 0] * u.mm, axes=['position_x'])
    position_y = kgpy.labeled.Array([0] * u.mm, axes=['position_y'])
    wavelength = kgpy.labeled.Array([63] * u.nm, axes=['wavelength'])
    efficiency = kgpy.labeled.Array(arr[..., 1] * u.dimensionless_unscaled, axes=['position_x'])
    return kgpy.function.Array(
        input=kgpy.optics.vectors.SpectralPositionVector(
            position=kgpy.vectors.Cartesian2D(position_x, position_y),
            wavelength=wavelength,
        ),
        output=efficiency,
    )


def vs_position_y() -> kgpy.function.Array[kgpy.optics.vectors.SpectralPositionVector, kgpy.labeled.Array]:
    df = pandas.read_table(file_vs_position_y)
    arr = df.to_numpy()
    position_x = kgpy.labeled.Array([0] * u.mm, axes=['position_x'])
    position_y = kgpy.labeled.Array(arr[..., 0] * u.mm, axes=['position_y'])
    wavelength = kgpy.labeled.Array([63] * u.nm, axes=['wavelength'])
    efficiency = kgpy.labeled.Array(arr[..., 1] * u.dimensionless_unscaled, axes=['position_y'])
    return kgpy.function.Array(
        input=kgpy.optics.vectors.SpectralPositionVector(
            # position=kgpy.vectors.Cartesian2D(position_x, position_y),
            wavelength=wavelength,
            position_x=position_x,
            position_y=position_y,
        ),
        output=efficiency,
    )


def vs_wavelength() -> kgpy.function.Array[kgpy.optics.vectors.InputAngleVector, kgpy.labeled.Array]:
    df = pandas.read_table(file_vs_wavelength, sep=' ')
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

