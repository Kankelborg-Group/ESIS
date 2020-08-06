import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
from kgpy import Name, optics
from .. import Component

__all__ = ['Detector']

AperSurfT = optics.surface.Standard[None, optics.aperture.Rectangular]
MainSurfT = optics.surface.Standard[None, optics.aperture.Rectangular]


@dataclasses.dataclass
class Detector(Component):
    name: Name = dataclasses.field(default_factory=lambda: Name('detector'))
    piston: u.Quantity = 0 * u.mm
    channel_radius: u.Quantity = 0 * u.mm
    channel_angle: u.Quantity = 0 * u.deg
    inclination: u.Quantity = 0 * u.deg
    pix_half_width_x: u.Quantity = 0 * u.mm
    pix_half_width_y: u.Quantity = 0 * u.mm
    npix_x: int = 0
    npix_y: int = 0
    border_width_right: u.Quantity = 0 * u.mm
    border_width_left: u.Quantity = 0 * u.mm
    border_width_top: u.Quantity = 0 * u.mm
    border_width_bottom: u.Quantity = 0 * u.mm

    @property
    def surface(self) -> AperSurfT:
        return optics.surface.Standard(
            name=self.name + 'aper',
            aperture=optics.aperture.Rectangular(
                half_width_x=self.npix_x * self.pix_half_width_x,
                half_width_y=self.npix_y * self.pix_half_width_y,
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
                    aperture=optics.aperture.AsymmetricRectangular(
                        is_active=False,
                        width_x_neg=self.border_width_left,
                        width_x_pos=self.border_width_right,
                        width_y_neg=self.border_width_bottom,
                        width_y_pos=self.border_width_top,
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
                    tilt=optics.coordinate.Tilt(y=self.inclination),
                ),
            ],
            is_last_surface=True,
        )

    def copy(self) -> 'Detector':
        return Detector(
            piston=self.piston.copy(),
            channel_radius=self.channel_radius.copy(),
            channel_angle=self.channel_angle.copy(),
            inclination=self.inclination.copy(),
            pix_half_width_x=self.pix_half_width_x.copy(),
            pix_half_width_y=self.pix_half_width_y.copy(),
            npix_x=self.npix_x,
            npix_y=self.npix_y,
            border_width_right=self.border_width_right.copy(),
            border_width_left=self.border_width_left.copy(),
            border_width_top=self.border_width_top.copy(),
            border_width_bottom=self.border_width_bottom.copy(),
            name=self.name.copy()
        )
