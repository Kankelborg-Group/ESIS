import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika.mixins
from . import mixins

__all__ = [
    "Grating",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractGrating(
    optika.mixins.Printable,
    optika.mixins.Rollable,
    optika.mixins.Yawable,
    optika.mixins.Pitchable,
    optika.mixins.Translatable,
    mixins.CylindricallyTransformable,
):
    @property
    @abc.abstractmethod
    def serial_number(self) -> str:
        """serial number of this diffraction grating"""

    @property
    @abc.abstractmethod
    def manufacturing_number(self) -> str:
        """additional number describing this diffraction grating"""

    @property
    @abc.abstractmethod
    def angle_input(self) -> u.Quantity | na.AbstractScalar:
        """the nominal angle of incident light"""

    @property
    @abc.abstractmethod
    def angle_output(self) -> u.Quantity | na.AbstractScalar:
        """the nominal angle of reflected light"""

    @property
    @abc.abstractmethod
    def sag(self) -> None | optika.sags.AbstractSag:
        """the sag function of this grating"""

    @property
    @abc.abstractmethod
    def material(self) -> None | optika.materials.AbstractMaterial:
        """optical material of this grating"""

    @property
    @abc.abstractmethod
    def rulings(self) -> None | optika.rulings.AbstractRulings:
        """ruling pattern of this grating"""

    @property
    @abc.abstractmethod
    def num_folds(self) -> int:
        """
        The order of the rotational symmetry of the optical system.
        This determines the aperture wedge angle of this grating.
        """

    @property
    def angle_aperture(self) -> u.Quantity | na.AbstractScalar:
        return (360 * u.deg) / self.num_folds

    @property
    @abc.abstractmethod
    def halfwidth_inner(self) -> u.Quantity | na.AbstractScalar:
        """distance from the apex to the inner edge of the clear aperture"""

    @property
    @abc.abstractmethod
    def halfwidth_outer(self) -> u.Quantity | na.AbstractScalar:
        """distance from the apex to the outer edge of the clear aperture"""

    @property
    @abc.abstractmethod
    def width_border(self) -> u.Quantity | na.AbstractScalar:
        """nominal width of the border around the clear aperture"""

    @property
    @abc.abstractmethod
    def width_border_inner(self) -> u.Quantity | na.AbstractScalar:
        """
        width of the border between the inner edge of the clear aperture
        and the substrate inner edge of the substrate.
        """

    @property
    @abc.abstractmethod
    def clearance(self) -> u.Quantity | na.AbstractScalar:
        """minimum distance between adjacent physical gratings"""

    @property
    def surface(self) -> optika.surfaces.Surface:
        angle_aperture = self.angle_aperture
        halfwidth_inner = self.halfwidth_inner
        halfwidth_outer = self.halfwidth_outer
        width_border = self.width_border
        width_border_inner = self.width_border_inner
        clearance = self.clearance
        distance_radial = self.distance_radial
        side_border_x = width_border / np.sin(angle_aperture / 2) + clearance
        offset_clear = distance_radial - side_border_x
        offset_mechanical = distance_radial - clearance
        return optika.surfaces.Surface(
            name="grating",
            sag=self.sag,
            material=self.material,
            aperture=optika.apertures.IsoscelesTrapezoidalAperture(
                x_left=offset_clear - halfwidth_inner,
                x_right=offset_clear + halfwidth_outer,
                angle=angle_aperture,
                transformation=na.transformations.Cartesian3dTranslation(
                    x=-offset_clear,
                ),
            ),
            aperture_mechanical=optika.apertures.IsoscelesTrapezoidalAperture(
                x_left=offset_mechanical - (halfwidth_inner + width_border_inner),
                x_right=offset_mechanical + halfwidth_outer + width_border,
                angle=angle_aperture,
                transformation=na.transformations.Cartesian3dTranslation(
                    x=-offset_mechanical,
                ),
            ),
            rulings=self.rulings,
            is_pupil_stop=True,
            transformation=self.transformation,
        )


@dataclasses.dataclass(eq=False, repr=False)
class Grating(
    AbstractGrating,
):
    serial_number: str = ""
    manufacturing_number: str = ""
    angle_input: u.Quantity = 0 * u.deg
    angle_output: u.Quantity = 0 * u.deg
    sag: None | optika.sags.AbstractSag = None
    material: None | optika.materials.AbstractMaterial = None
    rulings: None | optika.rulings.AbstractRulings = None
    num_folds: int = 0
    halfwidth_inner: u.Quantity | na.AbstractScalar = 0 * u.mm
    halfwidth_outer: u.Quantity | na.AbstractScalar = 0 * u.mm
    width_border: u.Quantity | na.AbstractScalar = 0 * u.mm
    width_border_inner: u.Quantity | na.AbstractScalar = 0 * u.mm
    clearance: u.Quantity | na.AbstractScalar = 0 * u.mm
    distance_radial: u.Quantity | na.AbstractScalar = 0 * u.mm
    azimuth: u.Quantity | na.AbstractScalar = 0 * u.deg
    translation: u.Quantity | na.AbstractCartesian3dVectorArray = 0 * u.mm
    pitch: u.Quantity | na.AbstractScalar = 0 * u.deg
    yaw: u.Quantity | na.AbstractScalar = 0 * u.deg
    roll: u.Quantity | na.AbstractScalar = 0 * u.deg
