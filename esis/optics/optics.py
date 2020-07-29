import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
from kgpy import Name, optics, vector
from . import components as cmps

__all__ = ['Optics']

default_name = Name('ESIS')


@dataclasses.dataclass
class Optics:
    name: Name = dataclasses.field(default_factory=lambda: default_name)
    components: cmps.Components = dataclasses.field(default_factory=lambda: cmps.Components())
    wavelengths: u.Quantity = 0 * u.nm
    field_limit: u.Quantity = 0 * u.deg
    pupil_samples: int = 10
    field_samples: int = 10

    def __post_init__(self):
        self.update()

    def update(self) -> typ.NoReturn:
        self._system = None

    @property
    def system(self) -> optics.System:
        if self._system is None:
            self._system = self._calc_system()
        return self._system

    def _calc_system(self) -> optics.System:
        return optics.System(
            object_surface=optics.surface.ObjectSurface(thickness=np.inf * u.mm),
            surfaces=self.components,
            stop_surface=self.components.grating.surface,
            wavelengths=self.wavelengths,
            pupil_samples=self.pupil_samples,
            field_min=-self.field_limit,
            field_max=self.field_limit,
            field_samples=self.field_samples,
        )
