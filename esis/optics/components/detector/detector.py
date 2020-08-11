import typing as typ
import dataclasses
import numpy as np
import pandas
import astropy.units as u
from kgpy import Name, optics, format
from .. import Component, Grating

__all__ = ['Detector']

AperSurfT = optics.surface.Standard[None, optics.aperture.Rectangular]
MainSurfT = optics.surface.Standard[None, optics.aperture.AsymmetricRectangular]


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
    dynamic_clearance: u.Quantity = 0 * u.mm

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
    def main_surface(self) -> MainSurfT:
        aper = self.surface.aperture
        return optics.surface.Standard(
            name=self.name + 'main',
            aperture=optics.aperture.AsymmetricRectangular(
                is_active=False,
                width_x_neg=-(aper.half_width_x + self.border_width_left),
                width_x_pos=aper.half_width_x + self.border_width_right,
                width_y_neg=-(aper.half_width_y + self.border_width_bottom),
                width_y_pos=aper.half_width_y + self.border_width_top,
            )
        )

    @property
    def _surfaces(self) -> optics.surface.Transformed[optics.surface.Substrate[AperSurfT, MainSurfT]]:
        return optics.surface.Transformed(
            name=self.name,
            surfaces=optics.surface.Substrate(
                aperture_surface=self.surface,
                main_surface=self.main_surface,
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
        return type(self)(
            name=self.name.copy(),
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
            dynamic_clearance=self.dynamic_clearance.copy(),
        )

    @property
    def dataframe(self) -> pandas.DataFrame:
        return pandas.DataFrame.from_dict(
            data={
                'piston': format.quantity(self.piston.to(u.mm)),
                'channel radius': format.quantity(self.channel_radius.to(u.mm)),
                'channel angle': format.quantity(self.channel_angle.to(u.deg)),
                'inclination': format.quantity(self.inclination.to(u.deg)),
                'pixel x half-width': format.quantity(self.pix_half_width_x.to(u.um)),
                'pixel y half-width': format.quantity(self.pix_half_width_y.to(u.um)),
                'number of pixels along x': self.npix_x,
                'number of pixels along y': self.npix_y,
                'right border width': format.quantity(self.border_width_right.to(u.mm)),
                'left border width': format.quantity(self.border_width_left.to(u.mm)),
                'top border width': format.quantity(self.border_width_top.to(u.mm)),
                'bottom border width': format.quantity(self.border_width_bottom.to(u.mm)),
                'dynamic clearance': format.quantity(self.dynamic_clearance.to(u.mm)),
            },
            orient='index',
            columns=[str(self.name)],
        )

    def apply_poletto_prescription(
            self,
            wavelength_1: u.Quantity,
            wavelength_2: u.Quantity,
            magnification: u.Quantity,
            primary_focal_length: u.Quantity,
            grating: Grating,
    ) -> 'Detector':
        if not grating.is_toroidal and grating.is_vls:
            wavelength = (wavelength_1 + wavelength_2) / 2
            f = primary_focal_length
            M = magnification
            alpha = grating.nominal_input_angle
            beta = grating.diffraction_angle(wavelength, alpha)
            r_A = np.sqrt(np.square(grating.channel_radius) + np.square(grating.piston - f))
            r_B = M * r_A
            R = grating.tangential_radius
            x_g = grating.piston
            r_g = grating.channel_radius

            sin_alpha, cos_alpha, tan_alpha = np.sin(alpha), np.cos(alpha), np.tan(alpha)
            sin_beta, cos_beta, tan_beta = np.sin(beta), np.cos(beta), np.tan(beta)

            tanphi_1spec = r_B * tan_beta / (R * cos_beta) - tan_beta
            tanphi_1spat = r_B * sin_beta / R
            tanphi_2spec = r_B * (tan_beta - tan_alpha) / (R * cos_beta) - M * r_g * cos_alpha / ((x_g - f) * cos_beta)
            tanphi_2spat = r_B * (sin_beta - tan_alpha * cos_beta) / R - M * r_g * cos_beta / ((x_g - f) * cos_alpha)
            phi = np.arctan([tanphi_1spec, tanphi_1spat, tanphi_2spec, tanphi_2spat]) << u.rad
            phi_avg = (np.max(phi) + np.min(phi)) / 2  # compromise value for detector tilt

            other = self.copy()
            other.piston = grating.piston - r_B * np.cos(beta + grating.inclination)
            other.inclination = (grating.inclination + beta) - phi_avg

            return other

        else:
            raise ValueError('Only SVLS supported')
