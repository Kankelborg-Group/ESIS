import abc
import dataclasses
import astropy.units as u
import named_arrays as na
import optika
from . import mixins

__all__ = [
    "Filter",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractFilter(
    optika.mixins.Printable,
    optika.mixins.Rollable,
    optika.mixins.Yawable,
    optika.mixins.Pitchable,
    optika.mixins.Translatable,
    mixins.CylindricallyTransformable,
):
    @property
    @abc.abstractmethod
    def material(self) -> optika.materials.AbstractThinFilmFilter:
        """A model of the filter material including the mesh and oxide."""

    @property
    @abc.abstractmethod
    def radius_clear(self) -> u.Quantity | na.AbstractScalar:
        """radius of the circular clear aperture"""

    @property
    @abc.abstractmethod
    def width_border(self) -> u.Quantity | na.AbstractScalar:
        """width of the frame around the clear aperture"""

    @property
    def surface(self) -> optika.surfaces.Surface:

        radius_clear = self.radius_clear
        radius_mech = radius_clear + self.width_border
        aperture = optika.apertures.CircularAperture(radius_clear)
        aperture_mechanical = optika.apertures.CircularAperture(radius_mech)

        return optika.surfaces.Surface(
            name="filter",
            material=self.material,
            aperture=aperture,
            aperture_mechanical=aperture_mechanical,
            transformation=self.transformation,
        )


@dataclasses.dataclass(eq=False, repr=False)
class Filter(
    AbstractFilter,
):
    material: None | optika.materials.AbstractMaterial = None
    material_oxide: None | optika.materials.AbstractMaterial = None
    material_mesh: None | optika.materials.AbstractMaterial = None
    ratio_mesh: float | na.AbstractScalar = 0
    frequency_mesh: u.Quantity | na.AbstractScalar = 0 / u.mm
    radius_clear: u.Quantity | na.AbstractScalar = 0 * u.mm
    width_border: u.Quantity | na.AbstractScalar = 0 * u.mm
    thickness: u.Quantity | na.AbstractScalar = 0 * u.mm
    thickness_oxide: u.Quantity | na.AbstractScalar = 0 * u.mm
    distance_radial: u.Quantity | na.AbstractScalar = 0 * u.mm
    azimuth: u.Quantity | na.AbstractScalar = 0 * u.deg
    translation: u.Quantity | na.AbstractCartesian3dVectorArray = 0 * u.mm
    pitch: u.Quantity | na.AbstractScalar = 0 * u.deg
    yaw: u.Quantity | na.AbstractScalar = 0 * u.deg
    roll: u.Quantity | na.AbstractScalar = 0 * u.deg
