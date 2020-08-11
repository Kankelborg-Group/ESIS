import typing as typ
import abc
import kgpy.optics
import kgpy.mixin

__all__ = ['Component']


class Component(kgpy.mixin.Named, abc.ABC):

    @property
    @abc.abstractmethod
    def _surfaces(self) -> typ.Iterable[kgpy.optics.Surface]:
        pass

    @property
    @abc.abstractmethod
    def surface(self) -> kgpy.optics.Surface:
        pass

    def __iter__(self) -> typ.Iterator[kgpy.optics.Surface]:
        return self._surfaces.__iter__()

    def copy(self) -> 'Component':
        pass
