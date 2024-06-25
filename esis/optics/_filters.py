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
    def material(self) -> None | optika.materials.AbstractMaterial:
        """the nominal material of the filter"""

    @property
    @abc.abstractmethod
    def material_oxide(self) -> None | optika.materials.AbstractMaterial:
        """material representing the oxidation layer"""

    @property
    @abc.abstractmethod
    def material_mesh(self) -> None | optika.materials.AbstractMaterial:
        """name of the material that the mesh is made from"""

    @property
    @abc.abstractmethod
    def ratio_mesh(self) -> float | na.AbstractScalar:
        """the fraction of light that the mesh blocks"""

    @property
    @abc.abstractmethod
    def frequency_mesh(self) -> u.Quantity | na.AbstractScalar:
        """the number of mesh lines per unit length"""

    @property
    @abc.abstractmethod
    def radius_clear(self) -> u.Quantity | na.AbstractScalar:
        """radius of the circular clear aperture"""

    @property
    @abc.abstractmethod
    def width_border(self) -> u.Quantity | na.AbstractScalar:
        """width of the frame around the clear aperture"""

    @property
    @abc.abstractmethod
    def thickness(self) -> u.Quantity | na.AbstractScalar:
        """nominal physical thickness of the filter"""

    @property
    @abc.abstractmethod
    def thickness_oxide(self) -> u.Quantity | na.AbstractScalar:
        """thickness of the oxide layers on the outside of the filter"""

    @property
    def surfaces(self) -> list[optika.surfaces.Surface]:
        material = self.material
        material_oxide = self.material_oxide
        radius_clear = self.radius_clear
        radius_mech = radius_clear + self.width_border
        aperture = optika.apertures.CircularAperture(radius_clear)
        aperture_mechanical = optika.apertures.CircularAperture(radius_mech)
        thickness = self.thickness
        thickness_oxide = self.thickness_oxide
        t = self.transformation
        return [
            optika.surfaces.Surface(
                name="filter-oxide-front",
                material=material_oxide,
                aperture=aperture,
                aperture_mechanical=aperture_mechanical,
                transformation=t,
            ),
            optika.surfaces.Surface(
                name="filter-front",
                material=material,
                transformation=t
                @ na.transformations.Cartesian3dTranslation(
                    z=thickness_oxide,
                ),
            ),
            optika.surfaces.Surface(
                name="filter-back",
                material=material_oxide,
                transformation=t
                @ na.transformations.Cartesian3dTranslation(
                    z=thickness_oxide + thickness
                ),
            ),
            optika.surfaces.Surface(
                name="filter-oxide-back",
                aperture=aperture,
                aperture_mechanical=aperture_mechanical,
                transformation=t
                @ na.transformations.Cartesian3dTranslation(
                    z=thickness_oxide + thickness + thickness_oxide
                ),
            ),
        ]


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
