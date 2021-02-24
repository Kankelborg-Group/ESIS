from dataclasses import dataclass
import typing as typ
import pathlib

__all__ = ['main_event', 'perfectx', 'loopflow', 'otherx', 'mgxfree']

@dataclass
class Event:
    name: str
    location: typ.Tuple[slice,slice]


    @property
    def mart_inverted_pickle_path(self):
        return pathlib.Path(__file__).parent / ('lev4_' + self.name + '_mart.pickle')


# A storage of interesting events in the 2019 flight, stored in pixels of the Level 3 data (not ideal but good for now)

main_event = Event('mainevent',(slice(500,600),slice(640,740)))

perfectx = Event('perfectx',(slice(540, 620), slice(1010, 1090)))

loopflow = Event('loopflow',(slice(310, 380), slice(835, 890)))

otherx = Event('otherx', (slice(420, 520), slice(840, 920)))

mgxfree = Event('mgxfree',(slice(710-10, 975+10), slice(585-10, 755+10)))
