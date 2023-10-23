from __future__ import annotations
import abc
import dataclasses
import functools
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
    optika.mixins.Rollable,
    optika.mixins.Yawable,
    optika.mixins.Pitchable,
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
        `diffraction grating equation <https://en.wikipedia.org/wiki/Diffraction_grating>`_.
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

    @property
    def _wavelength_test_grid(self) -> na.AbstractScalar:
        position = na.Cartesian3dVectorArray() * u.mm
        grating = self.grating.surface
        m = grating.rulings.diffraction_order
        d = grating.rulings.spacing(position)
        a = self.angle_grating_input
        b = self.angle_grating_output
        result = (np.sin(a) + np.sin(b)) * d / m
        return result.to(u.AA)

    @property
    def wavelength_min(self) -> u.Quantity | na.AbstractScalar:
        """the minimum wavelength permitted through the system"""
        return self._wavelength_test_grid.min(
            axis=("wire_grating_input", "wire_grating_output"),
        )

    @property
    def wavelength_max(self) -> u.Quantity | na.AbstractScalar:
        """the maximum wavelength permitted through the system"""
        return self._wavelength_test_grid.max(
            axis=("wire_grating_input", "wire_grating_output"),
        )

    @functools.cached_property
    def system(self) -> optika.systems.SequentialSystem:
        """
        Resolve this optics model into an instance of
        :class:`optika.systems.SequentialSystem`.

        This is a cached property that is only computed once.
        """
        surfaces = []
        surfaces += [self.front_aperture.surface]
        surfaces += [self.central_obscuration.surface]
        surfaces += [self.primary_mirror.surface]
        surfaces += [self.field_stop.surface]
        surfaces += [self.grating.surface]
        surfaces += self.filter.surfaces if self.filter is not None else []
        surfaces += [self.detector.surface]

        wavelength_min = self.wavelength_min
        wavelength_max = self.wavelength_max
        wavelength_range = wavelength_max - wavelength_min
        grid = self.grid_input_normalized
        grid.wavelength = wavelength_range * (grid.wavelength + 1) / 2 + wavelength_min

        result = optika.systems.SequentialSystem(
            surfaces=surfaces,
            grid_input_normalized=grid,
            transformation=self.transformation,
        )

        return result


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
    pitch: u.Quantity | na.AbstractScalar = 0 * u.deg
    yaw: u.Quantity | na.AbstractScalar = 0 * u.deg
    roll: u.Quantity | na.AbstractScalar = 0 * u.deg
