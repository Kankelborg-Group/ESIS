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
    gain_tap1: u.Quantity = 0 * u.electron / u.ct
    gain_tap2: u.Quantity = 0 * u.electron / u.ct
    gain_tap3: u.Quantity = 0 * u.electron / u.ct
    gain_tap4: u.Quantity = 0 * u.electron / u.ct
    readout_noise_tap1: u.Quantity = 0 * u.ct
    readout_noise_tap2: u.Quantity = 0 * u.ct
    readout_noise_tap3: u.Quantity = 0 * u.ct
    readout_noise_tap4: u.Quantity = 0 * u.ct

    @property
    def num_pixels_all(self) -> typ.Tuple[int,int]:
        return (self.num_pixels[vector.ix]+ 2 * self.npix_overscan + 2 * self.npix_blank , self.num_pixels[vector.iy] )

    @property
    def quadrants(self) -> typ.Tuple[typ.Tuple[slice, slice], ...]:
        half_height = self.num_pixels_all[vector.ix] // 2
        half_width = self.num_pixels_all[vector.iy] // 2
        quad_1 = slice(half_height), slice(half_width)
        quad_2 = slice(half_height, None), slice(half_width)
        quad_3 = slice(half_height, None), slice(half_width, None)
        quad_4 = slice(half_height), slice(half_width, None)
        return quad_1, quad_2, quad_3, quad_4

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

    @property
    def gain(self) -> u.Quantity:
        return u.Quantity([self.gain_tap1, self.gain_tap2, self.gain_tap3, self.gain_tap4])

    def dn_to_photon(self, data: u.Quantity, wavelength: u.Quantity) -> u.Quantity:
        """
        For a given wavelength return a detector size array with units of photon / DN
        """
        photon_data = data.value * u.photon

        channel_gains = self.gain
        quadrants = self.quadrants
        photon_energy = const.c * const.h / wavelength / u.photon
        Si_e_hole_pair = 3.6 * u.eV / u.electron

        print(channel_gains)
        for channel,channel_gain in enumerate(channel_gains):
            for i, quad in enumerate(quadrants):
                photon_data[(...,channel)+quad] = data[(...,channel)+quad] * channel_gain[i] * Si_e_hole_pair / photon_energy
        return photon_data

    def remove_inactive_pixels(self, frames: np.ndarray, axis: int = ~0):
        frames = self.remove_overscan_pixels(frames, ccd_long_axis=axis)
        frames = self.remove_blank_pixels(frames, axis=axis)

        return frames

    def remove_blank_pixels(self, frames: np.ndarray, axis: int = ~0):
        s = self.identify_blank_pixels(frames, axis)

        return frames[s]

    def identify_blank_pixels(self, frames: np.ndarray, axis: int = ~0):
        s = [slice(None)] * frames.ndim
        s[-1] = slice(self.npix_blank, ~(self.npix_blank - 1))
        s = tuple(s)

        return s

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
        new_half_len = half_len - self.npix_overscan

        s0[ccd_long_axis] = slice(None, new_half_len)
        s1[ccd_long_axis] = slice(~(new_half_len - 1), None)

        s0 = tuple(s0)
        s1 = tuple(s1)

        return s0, s1

    def remove_overscan_pixels(self,frames: np.ndarray, ccd_long_axis: int = ~0):
        """
        Trim the overscan pixels from an array of ESIS images.
        The overscan pixels are in the center of the images, running perpendicular to the long axis of the CCD.
        They are the last pixels to be read out on each row of each quadrant.
        :param frames: A sequence of ESIS images
        :param n_overscan_pix: The number of overscan pixels to remove from each quadrant.
        :param ccd_long_axis: Axis index of the CCD's long axis.
        :return: A copy of the `frames` array with the overscan pixels removed.
        """

        s0, s1 = self.identify_overscan_pixels(frames, ccd_long_axis)

        return np.concatenate([frames[s0], frames[s1]], axis=ccd_long_axis)

