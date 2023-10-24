import abc
import dataclasses
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "FrontAperture",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractFrontAperture(
    optika.mixins.Translatable,
):
    @property
    def surface(self) -> optika.surfaces.Surface:
        return optika.surfaces.Surface(
            name="front aperture",
            transformation=self.transformation,
        )


@dataclasses.dataclass(eq=False, repr=False)
class FrontAperture(
    AbstractFrontAperture,
):
    translation: u.Quantity | na.AbstractCartesian3dVectorArray = 0 * u.mm
