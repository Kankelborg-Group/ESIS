import typing as typ
import dataclasses
import kgpy.optics
from . import Component, DummySurface, FrontAperture, CentralObscuration, Primary, Filter, FieldStop, Grating, Detector

__all__ = ['Components']


@dataclasses.dataclass
class Components:

    dummy_surface: DummySurface = dataclasses.field(default_factory=lambda: DummySurface())
    front_aperture: FrontAperture = dataclasses.field(default_factory=lambda: FrontAperture())
    central_obscuration: CentralObscuration = dataclasses.field(default_factory=lambda: CentralObscuration())
    primary: Primary = dataclasses.field(default_factory=lambda: Primary())
    field_stop: FieldStop = dataclasses.field(default_factory=lambda: FieldStop())
    grating: Grating = dataclasses.field(default_factory=lambda: Grating())
    filter: Filter = dataclasses.field(default_factory=lambda: Filter())
    detector: Detector = dataclasses.field(default_factory=lambda: Detector())

    def __iter__(self) -> typ.Iterator[Component]:
        yield from self.dummy_surface
        yield from self.front_aperture
        yield from self.central_obscuration
        yield from self.primary
        yield from self.field_stop
        yield from self.grating
        yield from self.filter
        yield from self.detector
