import typing as typ
import dataclasses
import numpy as np
import pandas
import astropy.units as u
from kgpy import Name, vector, transform, optics, format, mixin
from . import Grating
from astropy import constants as const


__all__ = ['Detector']

SurfaceT = optics.surface.Surface[
    None,
    None,
    optics.surface.aperture.Rectangular,
    optics.surface.aperture.AsymmetricRectangular,
    None,
]


class DetectorAxes(mixin.AutoAxis):
    def __init__(self):
        super().__init__()
        self.detector_translation_x = self.auto_axis_index(from_right=False)
        self.detector_translation_y = self.auto_axis_index(from_right=False)
        self.detector_translation_z = self.auto_axis_index(from_right=False)


@dataclasses.dataclass
class Detector(optics.component.CylindricalComponent[SurfaceT]):
    name: Name = dataclasses.field(default_factory=lambda: Name('detector'))
    manufacturer: str = ''
    serial_number: np.ndarray = dataclasses.field(default_factory=lambda: np.array(''))
    range_focus_adjustment: u.Quantity = 0 * u.mm
    inclination: u.Quantity = 0 * u.deg
    roll: u.Quantity = 0 * u.deg
    twist: u.Quantity = 0 * u.deg
    pixel_width: u.Quantity = 0 * u.um
    num_pixels: typ.Tuple[int, int] = (0, 0)
    border_width_right: u.Quantity = 0 * u.mm
    border_width_left: u.Quantity = 0 * u.mm
    border_width_top: u.Quantity = 0 * u.mm
    border_width_bottom: u.Quantity = 0 * u.mm
    dynamic_clearance: u.Quantity = 0 * u.mm
    npix_overscan: int = 0
    npix_blank: int = 0
    temperature: u.Quantity = 0 * u.K
    gain: u.Quantity = 0 * u.electron / u.adu
    readout_noise: u.Quantity = 0 * u.adu
    dark_current: u.Quantity = 0 * u.electron / u.s
    charge_diffusion: u.Quantity = 0 * u.mm
    time_frame_transfer: u.Quantity = 0 * u.s
    time_readout: u.Quantity = 0 * u.s
    exposure_length: u.Quantity = 0 * u.s
    exposure_length_min: u.Quantity = 0 * u.s
    exposure_length_max: u.Quantity = 0 * u.s
    exposure_length_increment: u.Quantity = 0 * u.s
    bits_analog_to_digital: int = 0
    index_trigger: int = 0
    error_synchronization: u.Quantity = 0 * u.s
    position_ov: vector.Vector2D = dataclasses.field(default_factory=vector.Vector2D.spatial)

    @property
    def num_pixels_all(self) -> typ.Tuple[int, int]:
        """
        Number of pixels in each axis including the overscan and blank pixels

        Returns
        -------
        A tuple containing the total number of pixels along the long axis and short axis.
        """
        npix_x = self.num_pixels[vector.ix] + 2 * self.npix_overscan + 2 * self.npix_blank
        npix_y = self.num_pixels[vector.iy]
        return npix_x, npix_y

    @property
    def quadrants(self) -> typ.Tuple[typ.Tuple[slice, slice], ...]:
        """
        Slices for isolating the pixels associated with each tap on the CCD

        Returns
        -------
        A 4-tuple containing the pixel ranges for each axis of each quadrant on the sensor.
        """
        half_height = self.num_pixels_all[vector.iy] // 2
        half_width = self.num_pixels_all[vector.ix] // 2
        quad_1 = slice(half_height), slice(half_width)
        quad_2 = slice(half_height, None), slice(half_width)
        quad_3 = slice(half_height, None), slice(half_width, None)
        quad_4 = slice(half_height), slice(half_width, None)
        return quad_1, quad_2, quad_3, quad_4

    @property
    @u.quantity_input
    def pixel_half_width(self) -> u.um:
        return self.pixel_width / 2

    @property
    @u.quantity_input
    def clear_width(self) -> u.mm:
        return (self.num_pixels[vector.ix] * self.pixel_width).to(u.mm)

    @property
    @u.quantity_input
    def clear_height(self) -> u.mm:
        return (self.num_pixels[vector.iy] * self.pixel_width).to(u.mm)

    @property
    @u.quantity_input
    def clear_half_width(self) -> u.mm:
        return self.clear_width / 2

    @property
    @u.quantity_input
    def clear_half_height(self) -> u.mm:
        return self.clear_height / 2

    @property
    def transform(self) -> transform.rigid.TransformList:
        return super().transform + transform.rigid.TransformList([
            transform.rigid.TiltZ(self.roll),
            transform.rigid.TiltY(self.inclination),
            transform.rigid.TiltX(self.twist),
        ])

    @property
    def surface(self) -> SurfaceT:
        surface = super().surface
        surface.aperture = optics.surface.aperture.Rectangular(
            half_width_x=self.clear_half_width,
            half_width_y=self.clear_half_height,
        )
        surface.aperture_mechanical = optics.surface.aperture.AsymmetricRectangular(
            width_x_neg=-(self.clear_half_width + self.border_width_left),
            width_x_pos=self.clear_half_width + self.border_width_right,
            width_y_neg=-(self.clear_half_height + self.border_width_bottom),
            width_y_pos=self.clear_half_height + self.border_width_top,
        )
        surface.material = optics.surface.material.CCDStern2004()
        return surface

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        dataframe['manufacturer'] = [self.manufacturer]
        dataframe['focus adjustment range'] = [format.quantity(self.range_focus_adjustment)]
        dataframe['inclination'] = [format.quantity(self.inclination.to(u.deg))]
        dataframe['pixel width'] = [format.quantity(self.pixel_width.to(u.um))]
        dataframe['pixel array shape'] = [self.num_pixels]
        dataframe['right border width'] = [format.quantity(self.border_width_right.to(u.mm))]
        dataframe['left border width'] = [format.quantity(self.border_width_left.to(u.mm))]
        dataframe['top border width'] = [format.quantity(self.border_width_top.to(u.mm))]
        dataframe['bottom border width'] = [format.quantity(self.border_width_bottom.to(u.mm))]
        dataframe['dynamic clearance'] = [format.quantity(self.dynamic_clearance.to(u.mm))]
        dataframe['overscan pixels'] = [self.npix_overscan]
        dataframe['blank pixels'] = [self.npix_blank]
        dataframe['temperature'] = [format.quantity(self.temperature)]
        dataframe['gain'] = [format.quantity(self.gain)]
        dataframe['readout noise'] = [format.quantity(self.readout_noise)]
        dataframe['dark current'] = [format.quantity(self.dark_current)]
        dataframe['charge diffusion'] = [format.quantity(self.charge_diffusion)]
        dataframe['frame transfer time'] = [format.quantity(self.time_frame_transfer)]
        dataframe['readout time'] = [format.quantity(self.time_readout)]
        dataframe['exposure length'] = [format.quantity(self.exposure_length)]
        dataframe['minimum exposure length'] = [format.quantity(self.exposure_length_min)]
        dataframe['maximum exposure length'] = [format.quantity(self.exposure_length_max)]
        dataframe['exposure length increment'] = [format.quantity(self.exposure_length_increment)]
        dataframe['analog-to-digital bits'] = [self.bits_analog_to_digital]
        dataframe['trigger index'] = [self.index_trigger]
        dataframe['synchronization error'] = [format.quantity(self.error_synchronization)]
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

    # @property
    # def gain(self) -> u.Quantity:
    #     return u.Quantity([self.gain_tap1, self.gain_tap2, self.gain_tap3, self.gain_tap4])

    @u.quantity_input
    def convert_adu_to_electrons(self, data: u.adu) -> u.electron:
        """
        Converts data in ADU units to electron units using the gain for each tap in each channel

        Parameters
        ----------
        data
            sequence of images in ADU units

        Returns
        -------
        Sequence of images in electron units
        """
        data_new = np.empty(data.shape) * u.electron
        for c, gain_c in enumerate(self.gain):
            for q, quad in enumerate(self.quadrants):
                sl = (..., c) + quad
                data_new[sl] = data[sl] * gain_c[q]
        return data_new

    @u.quantity_input
    def convert_electrons_to_photons(self, data: u.electron, wavelength: u.nm) -> u.photon:
        """
        Converts data in electron units to photon units using the wavelength and the silicon workfunction

        Parameters
        ----------
        data
            sequence of images in electron units
        wavelength
            wavelength of incident light. Must be broadcastable with `data`

        Returns
        -------
        Sequence of images in photon units
        """
        workfunction = 3.6 * u.eV / u.electron
        photon_energy = const.c * const.h / wavelength / u.photon
        return data * workfunction / photon_energy

    @u.quantity_input
    def convert_adu_to_photons(self, data: u.adu, wavelength: u.nm) -> u.photon:
        """
        Converts data in ADU units to photon units.

        Parameters
        ----------
        data
            sequence of images in ADU units
        wavelength
            wavelength of incident light. Must be broadcastable with `data`

        Returns
        -------
        Sequence of images converted to photon units.
        """
        data = self.convert_adu_to_electrons(data)
        data = self.convert_electrons_to_photons(data, wavelength)
        return data

    @u.quantity_input
    def remove_inactive_pixels(self, data: u.adu) -> u.adu:
        """
        Given a raw image from the ESIS cameras, guass_fit_trim off the blank and overscan pixels, leaving only the light-
        sensitive pixels

        Parameters
        ----------
        data
            An array of raw ESIS images with the blank and overscan pixels still present
        Returns
        -------
        A cropped array containing only light-sensitive pixels
        """
        data_active = np.empty(data.shape[:~1] + self.num_pixels[::-1]) << data.unit
        half_width = self.num_pixels[vector.ix] // 2
        half_width_all = self.num_pixels_all[vector.ix] // 2
        data_active[..., :half_width] = data[..., self.npix_blank:half_width_all - self.npix_overscan]
        data_active[..., half_width:] = data[..., half_width_all + self.npix_overscan:-self.npix_blank]
        return data_active

    @u.quantity_input
    def readout_noise_image(self, num_channels: int) -> u.adu:
        """
        Calculate an image where each pixel contains the expected readout noise at that location

        Returns
        -------
        Image representing the readout noise at every pixel location.
        """
        readout_noise = np.broadcast_to(self.readout_noise, (num_channels, 4), subok=True)
        img = np.zeros((num_channels, ) + self.num_pixels_all[::-1]) << u.adu
        for c in range(num_channels):
            for q, quad in enumerate(self.quadrants):
                img[(..., c) + quad] = readout_noise[c, q]
        return img
