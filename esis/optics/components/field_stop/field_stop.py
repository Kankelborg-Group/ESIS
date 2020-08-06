import typing as typ
import dataclasses
import astropy.units as u
from kgpy import Name, optics
from .. import Component

__all__ = ['FieldStop']

AperSurfT = optics.surface.Standard[None, optics.aperture.RegularPolygon]
MainSurfT = optics.surface.Standard[None, optics.aperture.Circular]


@dataclasses.dataclass
class FieldStop(Component):
    name: Name = dataclasses.field(default_factory=lambda: Name('field_stop'))
    piston: u.Quantity = 0 * u.mm
    clear_radius: u.Quantity = 0 * u.mm
    mech_radius: u.Quantity = 0 * u.mm
    num_sides: int = 0

    @property
    def surface(self) -> AperSurfT:
        return optics.surface.Standard(
            name=self.name + 'aper',
            aperture=optics.aperture.RegularPolygon(
                is_test_stop=False,
                radius=self.clear_radius,
                num_sides=self.num_sides,
                offset_angle=180 * u.deg / self.num_sides,
            ),
        )

    @property
    def _surfaces(self) -> optics.surface.Transformed[optics.surface.Substrate[AperSurfT, MainSurfT]]:
        return optics.surface.Transformed(
            name=self.name,
            surfaces=optics.surface.Substrate(
                aperture_surface=self.surface,
                main_surface=optics.surface.Standard(
                    name=self.name + 'main',
                    aperture=optics.aperture.Circular(
                        is_active=False,
                        radius=self.mech_radius,
                    )
                )
            ),
            transforms=[
                optics.coordinate.Transform(
                    translate=optics.coordinate.Translate(z=-self.piston),
                )
            ],
        )

    def copy(self) -> 'FieldStop':
        return FieldStop(
            piston=self.piston.copy(),
            clear_radius=self.clear_radius.copy(),
            mech_radius=self.mech_radius.copy(),
            num_sides=self.num_sides,
            name=self.name.copy(),
        )
