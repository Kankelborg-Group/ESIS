import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "FieldStop",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractFieldStop(
    optika.mixins.Printable,
    optika.mixins.Translatable,
):
    @property
    @abc.abstractmethod
    def num_folds(self) -> int:
        """order of the rotational symmetry of the optical system"""

    @property
    def num_sides(self) -> int:
        """number of sides of the field stop's aperture"""
        return self.num_folds

    @property
    @abc.abstractmethod
    def radius_clear(self) -> u.Quantity | na.AbstractScalar:
        """distance from the center to a vertex of the clear aperture"""

    @property
    def width_clear(self) -> u.Quantity:
        return 2 * self.radius_clear * np.cos(360 * u.deg / self.num_sides / 2)

    @property
    @abc.abstractmethod
    def radius_mechanical(self) -> u.Quantity | na.AbstractScalar:
        """radius of the exterior edge of the field stop"""

    @property
    def surface(self) -> optika.surfaces.Surface:
        return optika.surfaces.Surface(
            name="field stop",
            aperture=optika.apertures.RegularPolygonalAperture(
                radius=self.radius_clear,
                num_vertices=self.num_sides,
            ),
            aperture_mechanical=optika.apertures.CircularAperture(
                radius=self.radius_mechanical,
            ),
            is_field_stop=True,
            transformation=self.transformation,
        )


@dataclasses.dataclass(eq=False, repr=False)
class FieldStop(
    AbstractFieldStop,
):
    num_folds: int = 0
    radius_clear: u.Quantity | na.AbstractScalar = 0 * u.mm
    radius_mechanical: u.Quantity | na.AbstractScalar = 0 * u.mm
    translation: u.Quantity | na.AbstractCartesian3dVectorArray = 0 * u.mm
