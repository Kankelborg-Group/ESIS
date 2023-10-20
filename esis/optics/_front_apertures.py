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
    @abc.abstractmethod
    def radius_clear(self) -> u.Quantity | na.AbstractScalar:
        """clear radius of the front aperture"""

    @property
    def surface(self) -> optika.surfaces.Surface:
        return optika.surfaces.Surface(
            name="front aperture",
            aperture=optika.apertures.CircularAperture(
                radius=self.radius_clear,
            ),
            transformation=self.transformation,
        )


@dataclasses.dataclass(eq=False, repr=False)
class FrontAperture(
    AbstractFrontAperture,
):
    radius_clear: u.Quantity | na.AbstractScalar = 0 * u.mm
    translation: u.Quantity | na.AbstractCartesian3dVectorArray = 0 * u.mm
