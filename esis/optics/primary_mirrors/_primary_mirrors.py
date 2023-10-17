import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika.mixins

__all__ = [
    "AbstractPrimaryMirror",
    "PrimaryMirror",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractPrimaryMirror(
    optika.mixins.Printable,
    optika.mixins.Transformable,
):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """human-readable name of this primary mirror"""

    @property
    @abc.abstractmethod
    def sag(self) -> None | optika.sags.AbstractSag:
        """the sag function of this primary mirror"""

    @property
    @abc.abstractmethod
    def num_sides(self) -> na.ScalarLike:
        """number of sides of the regular polygon"""

    @property
    @abc.abstractmethod
    def width_clear(self) -> na.ScalarLike:
        """width of the clear aperture from edge to edge"""

    @property
    def radius_clear(self) -> na.ScalarLike:
        """clear radius of the aperture from center to vertex"""
        halfwidth_clear = self.width_clear / 2
        num_sides = self.num_sides
        if (num_sides % 2) != 0:
            raise ValueError("odd numbers of sides not supported")
        result = halfwidth_clear / np.cos(360 * u.deg / num_sides / 2)
        return result

    @property
    @abc.abstractmethod
    def width_border(self) -> na.ScalarLike:
        """width of the border around the clear aperture"""

    @property
    def radius_mechanical(self) -> na.ScalarLike:
        """radius of the mechanical aperture from center to vertex"""
        halfwidth_clear = self.width_clear / 2
        width_border = self.width_border
        halfwidth = halfwidth_clear + width_border
        num_sides = self.num_sides
        if (num_sides % 2) != 0:
            raise ValueError("odd numbers of sides not supported")
        result = halfwidth / np.cos(360 * u.deg / num_sides / 2)
        return result

    @property
    @abc.abstractmethod
    def material(self) -> None | optika.materials.AbstractMaterial:
        """optical material of this component"""

    @property
    def surface(self) -> optika.surfaces.Surface:
        return optika.surfaces.Surface(
            name=self.name,
            sag=self.sag,
            material=self.material,
            aperture=optika.apertures.RegularPolygonalAperture(
                radius=self.radius_clear,
                num_vertices=self.num_sides,
            ),
            aperture_mechanical=optika.apertures.RegularPolygonalAperture(
                radius=self.radius_mechanical,
                num_vertices=self.num_sides,
            ),
            transformation=self.transformation,
        )


@dataclasses.dataclass(eq=False, repr=False)
class PrimaryMirror(
    AbstractPrimaryMirror,
):
    name: str = ""
    sag: None | optika.sags.AbstractSag = None
    num_sides: int = 0
    width_clear = 0 * u.mm
    width_border = 0 * u.mm
    material: None | optika.materials.AbstractMaterial = None
    transformation: None | na.transformations.AbstractTransformation = None
