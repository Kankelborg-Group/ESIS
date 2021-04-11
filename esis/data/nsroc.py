import dataclasses
import kgpy.nsroc

__all__ = ['Timeline']


@dataclasses.dataclass
class Timeline(kgpy.nsroc.Timeline):
    esis_exp_launch: kgpy.nsroc.Event = dataclasses.field(
        default_factory=lambda: kgpy.nsroc.Event(name=kgpy.Name('ESIS EXP launch')))

    def __iter__(self):
        yield from super().__iter__()
        yield self.esis_exp_launch
