import typing as typ
import dataclasses
from kgpy import mixin, optics
from .. import components as cmps

__all__ = ['Components']


@dataclasses.dataclass
class Components(mixin.Copyable):

    source: cmps.Source = dataclasses.field(default_factory=cmps.Source)
    front_aperture: cmps.FrontAperture = dataclasses.field(default_factory=cmps.FrontAperture)
    central_obscuration: cmps.CentralObscuration = dataclasses.field(default_factory=cmps.CentralObscuration)
    primary: cmps.Primary = dataclasses.field(default_factory=cmps.Primary)
    field_stop: cmps.FieldStop = dataclasses.field(default_factory=cmps.FieldStop)
    grating: cmps.Grating = dataclasses.field(default_factory=cmps.Grating)
    filter: cmps.Filter = dataclasses.field(default_factory=cmps.Filter)
    detector: cmps.Detector = dataclasses.field(default_factory=cmps.Detector)

    def __iter__(self) -> typ.Iterator[optics.Component]:
        yield self.front_aperture.surface
        yield self.central_obscuration.surface
        yield self.primary.surface
        yield self.field_stop.surface
        yield self.grating.surface
        yield self.filter.surface
        yield self.detector.surface

    def copy(self) -> 'Components':
        other = super().copy()  # type: Components
        other.source = self.source.copy()
        other.front_aperture = self.front_aperture.copy()
        other.central_obscuration = self.central_obscuration.copy()
        other.primary = self.primary.copy()
        other.field_stop = self.field_stop.copy()
        other.grating = self.grating.copy()
        other.filter = self.filter.copy()
        other.detector = self.detector.copy()
        return other
