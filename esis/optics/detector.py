import typing as typ
import dataclasses
import numpy as np
import pandas
import astropy.units as u
import kgpy.format
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vectors
import kgpy.transforms
import kgpy.optics
import kgpy.mixin
from . import Grating
from astropy import constants as const


__all__ = ['Detector']

SurfaceT = kgpy.optics.surfaces.Surface[
    None,
    None,
    kgpy.optics.surfaces.apertures.Rectangular,
    kgpy.optics.surfaces.apertures.Rectangular,
    None,
]


@dataclasses.dataclass
class Detector(kgpy.optics.components.CylindricalComponent[SurfaceT]):
    name: str = 'detector'
    manufacturer: str = ''
    serial_number: typ.Union[str, kgpy.labeled.Array[str]] = ''
    range_focus_adjustment: kgpy.labeled.ArrayLike = 0 * u.mm
    inclination: kgpy.uncertainty.ArrayLike = 0 * u.deg
    roll: kgpy.uncertainty.ArrayLike = 0 * u.deg
    twist: kgpy.uncertainty.ArrayLike = 0 * u.deg
    pixel_width: kgpy.labeled.ArrayLike = 0 * u.um
    num_pixels: kgpy.vectors.Cartesian2D[int, int] = dataclasses.field(default_factory=kgpy.vectors.Cartesian2D)
    border_width: kgpy.uncertainty.ArrayLike = 0 * u.mm
    dynamic_clearance: kgpy.labeled.ArrayLike = 0 * u.mm
    npix_overscan: int = 0
    npix_blank: int = 0
    temperature: kgpy.uncertainty.ArrayLike = 0 * u.K
    gain: kgpy.uncertainty.ArrayLike = 0 * u.electron / u.adu
    readout_noise: kgpy.uncertainty.ArrayLike = 0 * u.adu
    dark_current: kgpy.uncertainty.ArrayLike = 0 * u.electron / u.s
    charge_diffusion: kgpy.uncertainty.ArrayLike = 0 * u.mm
    time_frame_transfer: kgpy.uncertainty.ArrayLike = 0 * u.s
    time_readout: kgpy.uncertainty.ArrayLike = 0 * u.s
    exposure_length: kgpy.uncertainty.ArrayLike = 0 * u.s
    exposure_length_min: kgpy.labeled.ArrayLike = 0 * u.s
    exposure_length_max: kgpy.labeled.ArrayLike = 0 * u.s
    exposure_length_increment: kgpy.labeled.ArrayLike = 0 * u.s
    bits_analog_to_digital: int = 0
    index_trigger: int = 0
    error_synchronization: u.Quantity = 0 * u.s
    position_ov: kgpy.vectors.Cartesian2D = dataclasses.field(default_factory=lambda: kgpy.vectors.Cartesian2D() * u.mm)

    @property
    def num_pixels_all(self) -> kgpy.vectors.Cartesian2D:
        """
        Number of pixels in each axis including the overscan and blank pixels.
        """
        result = self.num_pixels.copy()
        result.x = result.x + 2 * self.npix_overscan + 2 * self.npix_blank
        return result

    @property
    def quadrants(self) -> typ.Tuple[typ.Dict[str, slice], ...]:
        """
        Slices for isolating the pixels associated with each tap on the CCD
        """
        npix = self.num_pixels_all
        pix = npix // 2
        quad_1 = dict(x=slice(pix.x), y=slice(pix.y))
        quad_2 = dict(x=slice(pix.x, None), y=slice(pix.y))
        quad_3 = dict(x=slice(pix.x, None), y=slice(pix.y, None))
        quad_4 = dict(x=slice(pix.x), y=slice(pix.y, None))
        return quad_1, quad_2, quad_3, quad_4

    @property
    def pixel_half_width(self) -> kgpy.labeled.ArrayLike:
        return self.pixel_width / 2

    @property
    def clear_width(self) -> kgpy.labeled.ArrayLike:
        return (self.num_pixels.x * self.pixel_width).to(u.mm)

    @property
    def clear_height(self) -> kgpy.labeled.ArrayLike:
        return (self.num_pixels.y * self.pixel_width).to(u.mm)

    @property
    def clear_half_width(self) -> kgpy.labeled.ArrayLike:
        return self.clear_width / 2

    @property
    def clear_half_height(self) -> kgpy.labeled.ArrayLike:
        return self.clear_height / 2

    @property
    def transform(self) -> kgpy.transforms.TransformList:
        return super().transform + kgpy.transforms.TransformList([
            kgpy.transforms.RotationZ(self.roll),
            kgpy.transforms.RotationY(self.inclination),
            kgpy.transforms.RotationX(self.twist),
        ])

    @property
    def surface(self) -> kgpy.optics.detectors.Detector:
        surface = kgpy.optics.detectors.Detector()
        surface.name = self.name
        surface.transform = self.transform
        surface.plot_kwargs = {**surface.plot_kwargs, **self.plot_kwargs}
        surface.aperture = kgpy.optics.surfaces.apertures.Rectangular(
            is_active=False,
            half_width=kgpy.vectors.Cartesian2D(self.clear_half_width, self.clear_half_height),
        )
        surface.aperture_mechanical = kgpy.optics.surfaces.apertures.Rectangular(
            half_width=kgpy.vectors.Cartesian2D(self.clear_half_width, self.clear_half_height) + self.border_width,
        )
        surface.material = kgpy.optics.surfaces.materials.CCDStern2004()
        surface.shape_pixels = self.num_pixels
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
