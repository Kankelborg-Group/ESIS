import abc
import dataclasses
import astropy.units as u
import named_arrays as na
import optika

__all__ = [
    "CylindricallyTransformable",
]


@dataclasses.dataclass(eq=False, repr=False)
class CylindricallyTransformable(
    optika.mixins.Transformable,
):
    @property
    @abc.abstractmethod
    def distance_radial(self) -> u.Quantity | na.AbstractScalar:
        """distance from the axis of symmetry"""

    @property
    @abc.abstractmethod
    def azimuth(self) -> u.Quantity | na.AbstractScalar:
        """angle of rotation about the axis of symmetry"""

    @property
    def transformation(self) -> na.transformations.AbstractTransformation:
        t = na.transformations.TransformationList(
            [
                na.transformations.Cartesian3dRotationZ(self.azimuth),
                na.transformations.Cartesian3dTranslation(x=self.distance_radial),
            ]
        )
        return super().transformation @ t
