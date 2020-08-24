import typing as typ
import dataclasses
import pandas
import astropy.units as u
from kgpy import Name, optics, format, transform
from . import Component

__all__ = ['CentralObscuration']

MainSurfT = optics.surface.Standard[None, optics.aperture.RegularPolygon]


@dataclasses.dataclass
class CentralObscuration(Component):
    name: Name = dataclasses.field(default_factory=lambda: Name('obscuration'))
    piston: u.Quantity = 0 * u.mm
    obscured_radius: u.Quantity = 0 * u.mm
    num_sides: int = 0

    @property
    def surface(self) -> MainSurfT:
        return optics.surface.Standard(
            name=self.name + 'main',
            aperture=optics.aperture.RegularPolygon(
                is_test_stop=False,
                is_obscuration=True,
                radius=self.obscured_radius,
                num_sides=self.num_sides,
                offset_angle=180 * u.deg / self.num_sides,
            ),
        )

    @property
    def _surfaces(self) -> optics.surface.Transformed[MainSurfT]:
        return optics.surface.Transformed(
            name=self.name,
            surfaces=self.surface,
            transform=transform.rigid.Translate.from_components(z=-self.piston),
        )

    def copy(self) -> 'CentralObscuration':
        return type(self)(
            piston=self.piston.copy(),
            obscured_radius=self.obscured_radius.copy(),
            num_sides=self.num_sides,
            name=self.name.copy(),
        )

    @property
    def dataframe(self) -> pandas.DataFrame:
        return pandas.DataFrame.from_dict(
            data={
                'piston': format.quantity(self.piston.to(u.mm)),
                'obscured radius': format.quantity(self.obscured_radius.to(u.mm)),
                'number of sides': self.num_sides,
            },
            orient='index',
            columns=[str(self.name)],
        )
