from __future__ import annotations
import abc
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
import optika
import esis

__all__ = [
    "OpticsModel",
]


@dataclasses.dataclass(eq=False, repr=False)
class AbstractOpticsModel(
    optika.mixins.Printable,
):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """human-readable name of the instrument"""

    @property
    @abc.abstractmethod
    def front_aperture(self) -> None | esis.optics.abc.AbstractFrontAperture:
        """model of the front aperture plate"""

    @property
    @abc.abstractmethod
    def central_obscuration(self) -> None | esis.optics.abc.AbstractCentralObscuration:
        """model of the central obscuration, which holds the diffraction grating array"""

    @property
    @abc.abstractmethod
    def primary_mirror(self) -> None | esis.optics.abc.AbstractPrimaryMirror:
        """model of the primary mirror"""

    @property
    @abc.abstractmethod
    def field_stop(self) -> None | esis.optics.abc.AbstractFieldStop:
        """model of the field stop"""

    @property
    @abc.abstractmethod
    def grating(self) -> None | esis.optics.abc.AbstractGrating:
        """model of the diffraction grating array"""

    @property
    @abc.abstractmethod
    def filter(self) -> None | esis.optics.abc.AbstractFilter:
        """model of the thin-film filters"""

    @property
    @abc.abstractmethod
    def detector(self) -> None | esis.optics.abc.AbstractDetector:
        """model of the CCD sensors"""

    @property
    @abc.abstractmethod
    def grid_input_normalized(self) -> None | optika.vectors.ObjectVectorArray:
        """Normalized wavelength, pupil, and field coordinates to sample the optical system with"""

    @property
    def angle_grating_input(self) -> na.AbstractScalar:
        fs = self.field_stop.surface
        grating = self.grating.surface
        position = na.Cartesian3dVectorArray() * u.mm
        normal_surface = grating.sag.normal(position)
        normal_rulings = grating.rulings.normal(position)
        transformation = grating.transformation.inverse @ fs.transformation
        wire = np.moveaxis(
            a=fs.aperture.wire(),
            source="wire",
            destination="wire_grating_input",
        )
        wire = transformation(wire)
        return np.arctan2(
            wire @ normal_rulings,
            wire @ normal_surface,
        )

    @property
    def angle_grating_output(self) -> na.AbstractScalar:
        """
        The angle between the grating normal vector and the exit arm,
        in the plane perpendicular to the rulings.

        This is an analogue to the diffracted angle in the
        `diffraction grating equation https://en.wikipedia.org/wiki/Diffraction_grating`_.
        """
        detector = self.detector.surface
        grating = self.grating.surface
        position = na.Cartesian3dVectorArray() * u.mm
        normal_surface = grating.sag.normal(position)
        normal_rulings = grating.rulings.normal(position)
        transformation = grating.transformation.inverse @ detector.transformation
        wire = np.moveaxis(
            a=detector.aperture.wire(),
            source="wire",
            destination="wire_grating_output",
        )
        wire = transformation(wire)
        return np.arctan2(
            wire @ normal_rulings,
            wire @ normal_surface,
        )


@dataclasses.dataclass(eq=False, repr=False)
class OpticsModel(
    AbstractOpticsModel,
):
    name: str = "ESIS"
    front_aperture: None | esis.optics.FrontAperture = None
    central_obscuration: None | esis.optics.CentralObscuration = None
    primary_mirror: None | esis.optics.PrimaryMirror = None
    field_stop: None | esis.optics.FieldStop = None
    grating: None | esis.optics.Grating = None
    filter: None | esis.optics.Filter = None
    detector: None | esis.optics.Detector = None
    grid_input_normalized: None | optika.vectors.ObjectVectorArray = None
