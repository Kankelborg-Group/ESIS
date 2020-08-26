import typing as typ
import dataclasses
import kgpy.optics
from .. import components as cmps

__all__ = ['Components']


@dataclasses.dataclass
class Components:

    dummy_surface: cmps.DummySurface = dataclasses.field(default_factory=cmps.DummySurface)
    front_aperture: cmps.FrontAperture = dataclasses.field(default_factory=cmps.FrontAperture)
    central_obscuration: cmps.CentralObscuration = dataclasses.field(default_factory=cmps.CentralObscuration)
    primary: cmps.Primary = dataclasses.field(default_factory=cmps.Primary)
    field_stop: cmps.FieldStop = dataclasses.field(default_factory=cmps.FieldStop)
    grating: cmps.Grating = dataclasses.field(default_factory=cmps.Grating)
    filter: cmps.Filter = dataclasses.field(default_factory=cmps.Filter)
    detector: cmps.Detector = dataclasses.field(default_factory=cmps.Detector)

    def __iter__(self) -> typ.Iterator[kgpy.optics.component.Component]:
        yield from self.dummy_surface
        yield from self.front_aperture
        yield from self.central_obscuration
        yield from self.primary
        yield from self.field_stop
        yield from self.grating
        yield from self.filter
        yield from self.detector
