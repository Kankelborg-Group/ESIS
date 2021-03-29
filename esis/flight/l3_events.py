from dataclasses import dataclass
import typing as typ
import pathlib
import matplotlib.patches as patches

__all__ = ['main_event', 'perfectx', 'loopflow', 'otherx', 'mgxfree', 'big_blue', 'little_red']


@dataclass
class Event:
    name: str
    location: typ.Tuple[slice, slice]

    @property
    def mart_inverted_pickle_path(self):
        return pathlib.Path(__file__).parent / ('lev4_' + self.name + '_mart.pickle')

    @property
    def rectangle(self):
        y_0 = self.location[0].start
        x_0 = self.location[1].start
        return patches.Rectangle((x_0, y_0),
                                 self.location[1].stop - x_0,
                                 self.location[0].stop - y_0,
                                 fill=False,
                                 color='r'
                                 )


# A storage of interesting events in the 2019 flight, stored in pixels of the Level 3 data (not ideal but good for now)

main_event = Event('mainevent', (slice(500, 600), slice(640, 740)))

x=6

perfectx = Event('perfectx', (slice(540+14+x+1, 620-14-x+1), slice(1010+14+x-1, 1090-14-x-1)))

big_blue = Event('big_blue', (slice(840+4-4+x,900-4-4-x), slice(144+x+2, 196-x+2)))

little_red = Event('little_red', (slice(450-15+4+x,480+15-4-x), slice(1150+2+4+x,2+1210-4-x)))

loopflow = Event('loopflow', (slice(310, 380), slice(835, 890)))
x=25
otherx = Event('otherx', (slice(420+x+8, 520-x-8), slice(840+x-2, 920-x-3)))

mgxfree = Event('mgxfree', (slice(710 - 10, 975 + 10), slice(585 - 10, 755 + 10)))

