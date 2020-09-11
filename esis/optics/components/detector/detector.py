import typing as typ
import dataclasses
import numpy as np
import pandas
import astropy.units as u
from kgpy import Name, vector, transform, optics, format
from .. import Grating
from astropy import constants as const


__all__ = ['Detector']

SurfaceT = optics.Surface[
    None,
    None,
    optics.aperture.Rectangular,
    optics.aperture.AsymmetricRectangular,
    None,
]


@dataclasses.dataclass
class Detector(optics.component.CylindricalComponent[SurfaceT]):
    name: Name = dataclasses.field(default_factory=lambda: Name('detector'))
    inclination: u.Quantity = 0 * u.deg
    pixel_width: u.Quantity = 0 * u.um
    num_pixels: typ.Tuple[int, int] = (0, 0)
    border_width_right: u.Quantity = 0 * u.mm
    border_width_left: u.Quantity = 0 * u.mm
    border_width_top: u.Quantity = 0 * u.mm
    border_width_bottom: u.Quantity = 0 * u.mm
    dynamic_clearance: u.Quantity = 0 * u.mm
    npix_overscan: int = 0
    npix_blank: int = 0
    gain_tap1: u.Quantity = 0 * u.electron
    gain_tap2: u.Quantity = 0 * u.electron
    gain_tap3: u.Quantity = 0 * u.electron
    gain_tap4: u.Quantity = 0 * u.electron
    readout_noise_tap1 = 0
    readout_noise_tap2 = 0
    readout_noise_tap3 = 0
    readout_noise_tap4 = 0




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
    def surface(self) -> SurfaceT:
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
        dataframe = super().dataframe
        dataframe['inclination'] = [format.quantity(self.inclination.to(u.deg))]
        dataframe['pixel width'] = [format.quantity(self.pixel_width.to(u.um))]
        dataframe['pixel array shape'] = [self.num_pixels]
        dataframe['right border width'] = [format.quantity(self.border_width_right.to(u.mm))]
        dataframe['left border width'] = [format.quantity(self.border_width_left.to(u.mm))]
        dataframe['top border width'] = [format.quantity(self.border_width_top.to(u.mm))]
        dataframe['bottom border width'] = [format.quantity(self.border_width_bottom.to(u.mm))]
        dataframe['dynamic clearance'] = [format.quantity(self.dynamic_clearance.to(u.mm))]
        return dataframe

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
            r_A = np.sqrt(np.square(grating.cylindrical_radius) + np.square(grating.piston - f))
            r_B = M * r_A
            R = grating.tangential_radius
            x_g = grating.piston
            r_g = grating.cylindrical_radius

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

    def dn_to_photon(self,data: np.ndarray, wavelength: u.Quantity) -> np.ndarray:
        """
        For a given wavelength return a detector size array with units of photon / DN
        """
        photon_energy = const.c * const.h / wavelength
        Si_e_hole_pair = 3.6 * u.eV

    @staticmethod
    def remove_inactive_pixels(frames: np.ndarray, n_overscan_pix, n_blank_pix, axis: int = ~0):
        frames = Level_1.remove_overscan_pixels(frames, n_overscan_pix, ccd_long_axis=axis)

        frames = Level_1.remove_blank_pixels(frames, n_blank_pix, axis=axis)

        return frames

    @staticmethod
    def remove_blank_pixels(frames: np.ndarray, n_blank_pixels: int, axis: int = ~0):
        s = Level_1.identify_blank_pixels(frames, n_blank_pixels, axis)

        return frames[s]

    @staticmethod
    def identify_blank_pixels(frames: np.ndarray, n_blank_pixels: int, axis: int = ~0):
        s = [slice(None)] * frames.ndim
        s[-1] = slice(n_blank_pixels, ~(n_blank_pixels - 1))
        s = tuple(s)

        return s

    @staticmethod
    def identify_overscan_pixels(
            self,
            frames: np.ndarray,
            ccd_long_axis: int = ~0
    ) -> typ.Tuple[typ.Tuple[typ.Union[slice, int], ...], ...]:
        """

        :param frames:
        :param n_overscan_pix:
        :param ccd_long_axis:
        :return:
        """
        s0 = [slice(None)] * frames.ndim
        s1 = [slice(None)] * frames.ndim

        half_len = frames.shape[ccd_long_axis] // 2
        new_half_len = half_len - self.n_overscan_pix

        s0[ccd_long_axis] = slice(None, new_half_len)
        s1[ccd_long_axis] = slice(~(new_half_len - 1), None)

        s0 = tuple(s0)
        s1 = tuple(s1)

        return s0, s1


    def remove_overscan_pixels(self,frames: np.ndarray int, ccd_long_axis: int = ~0):
        """
        Trim the overscan pixels from an array of ESIS images.
        The overscan pixels are in the center of the images, running perpendicular to the long axis of the CCD.
        They are the last pixels to be read out on each row of each quadrant.
        :param frames: A sequence of ESIS images
        :param n_overscan_pix: The number of overscan pixels to remove from each quadrant.
        :param ccd_long_axis: Axis index of the CCD's long axis.
        :return: A copy of the `frames` array with the overscan pixels removed.
        """

        s0, s1 = self.identify_overscan_pixels(self, frames, ccd_long_axis)

        return np.concatenate([frames[s0], frames[s1]], axis=ccd_long_axis)

