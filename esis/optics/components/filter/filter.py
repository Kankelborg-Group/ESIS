import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
from kgpy import Name, optics
from .. import Component

__all__ = ['Filter']

AperSurfT = optics.surface.Standard[None, optics.aperture.Circular]
MainSurfT = optics.surface.Standard[None, optics.aperture.Circular]


@dataclasses.dataclass
class Filter(Component):
    name: Name = dataclasses.field(default_factory=lambda: Name('filter'))
    piston: u.Quantity = 0 * u.mm
    channel_radius: u.Quantity = 0 * u .mm
    channel_angle: u.Quantity = 0 * u.deg
    inclination: u.Quantity = 0 * u.deg
    clear_radius: u.Quantity = 0 * u.mm
    border_width: u.Quantity = 0 * u.mm

    @property
    def surface(self) -> AperSurfT:
        return optics.surface.Standard(
            name=self.name + 'aper',
            aperture=optics.aperture.Circular(
                radius=self.clear_radius
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
                        radius=self.clear_radius + self.border_width,
                    )
                )
            ),
            transforms=[
                optics.coordinate.Transform(
                    translate=optics.coordinate.Translate(z=-self.piston)
                ),
                optics.coordinate.Transform(
                    tilt=optics.coordinate.Tilt(z=self.channel_angle),
                    translate=optics.coordinate.Translate(x=self.channel_radius),
                    tilt_first=True,
                ),
                optics.coordinate.Transform(
                    tilt=optics.coordinate.Tilt(y=-self.inclination),
                ),
            ]
        )

    def copy(self) -> 'Detector':
        return Filter(
            name=self.name.copy(),
            piston=self.piston.copy(),
            channel_radius=self.channel_radius.copy(),
            channel_angle=self.channel_angle.copy(),
            inclination=self.inclination.copy(),
            clear_radius=self.clear_radius.copy(),
            border_width=self.border_width.copy(),
        )
