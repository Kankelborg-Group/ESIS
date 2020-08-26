import typing as typ
import dataclasses
import numpy as np
import pandas
import astropy.units as u
from kgpy import Name, vector, transform, optics, format
from .. import Grating

__all__ = ['Detector']

SurfT = optics.surface.Standard[None, optics.aperture.Rectangular, optics.aperture.AsymmetricRectangular]


@dataclasses.dataclass
class Detector(optics.component.CylindricalComponent[SurfT]):
    name: Name = dataclasses.field(default_factory=lambda: Name('detector'))
    inclination: u.Quantity = 0 * u.deg
    pixel_width: u.Quantity = 0 * u.um
    num_pixels: typ.Tuple[int, int] = (0, 0)
    border_width_right: u.Quantity = 0 * u.mm
    border_width_left: u.Quantity = 0 * u.mm
    border_width_top: u.Quantity = 0 * u.mm
    border_width_bottom: u.Quantity = 0 * u.mm
    dynamic_clearance: u.Quantity = 0 * u.mm

    @property
    def pixel_half_width(self) -> u.Quantity:
        return self.pixel_width / 2

    @property
    def clear_width(self) -> u.Quantity:
        return (self.num_pixels[vector.ix] * self.pixel_width).to(u.mm)

    @property
    def clear_height(self) -> u.Quantity:
        return (self.num_pixels[vector.iy] * self.pixel_width).to(u.mm)

    @property
    def clear_half_width(self) -> u.Quantity:
        return self.clear_width / 2

    @property
    def clear_half_height(self) -> u.Quantity:
        return self.clear_height / 2

    @property
    def transform(self) -> transform.rigid.Transform:
        return super().transform + transform.rigid.TransformList([
            transform.rigid.TiltY(self.inclination),
        ])

    @property
    def surface(self) -> SurfT:
        surface = super().surface
        surface.aperture = optics.aperture.Rectangular(
            half_width_x=self.clear_half_width,
            half_width_y=self.clear_half_height,
        )
        surface.aperture_mechanical = optics.aperture.AsymmetricRectangular(
            width_x_neg=-(self.clear_half_width + self.border_width_left),
            width_x_pos=self.clear_half_width + self.border_width_right,
            width_y_neg=-(self.clear_half_height + self.border_width_bottom),
            width_y_pos=self.clear_half_height + self.border_width_top,
        )
        return surface

    def copy(self) -> 'Detector':
        other = super().copy()  # type: Detector
        other.inclination = self.inclination.copy()
        other.pixel_width = self.pixel_width.copy()
        other.num_pixels = self.num_pixels
        other.border_width_right = self.border_width_right.copy()
        other.border_width_left = self.border_width_left.copy()
        other.border_width_top = self.border_width_top.copy()
        other.border_width_bottom = self.border_width_bottom.copy()
        other.dynamic_clearance = self.dynamic_clearance.copy()
        return other

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
