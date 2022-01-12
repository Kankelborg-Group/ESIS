import typing as typ
import dataclasses
import numpy as np
import pandas
import scipy.optimize
import astropy.units as u
from kgpy import Name, optics, format, transform, mixin
from .. import poletto
from . import efficiency

__all__ = [
    'GratingAxes',
    'Grating',
]

SurfaceT = optics.surface.Surface[
    optics.surface.sag.Toroidal,
    optics.surface.material.MultilayerMirror,
    optics.surface.aperture.IsoscelesTrapezoid,
    optics.surface.aperture.IsoscelesTrapezoid,
    optics.surface.rulings.CubicPolyDensity,
]


class GratingAxes(mixin.AutoAxis):
    def __init__(self):
        super().__init__()
        self.grating_translation_x = self.auto_axis_index(from_right=True)
        self.grating_translation_y = self.auto_axis_index(from_right=True)
        self.grating_translation_z = self.auto_axis_index(from_right=True)
        self.grating_inclination = self.auto_axis_index(from_right=True)
        self.grating_roll = self.auto_axis_index(from_right=True)
        self.grating_twist = self.auto_axis_index(from_right=True)
        self.grating_tangential_radius = self.auto_axis_index(from_right=True)
        self.grating_sagittal_radius = self.auto_axis_index(from_right=True)
        self.grating_ruling_density = self.auto_axis_index(from_right=True)
        self.grating_ruling_spacing_coeff_linear = self.auto_axis_index(from_right=True)
        self.grating_ruling_spacing_coeff_quadratic = self.auto_axis_index(from_right=True)
        self.grating_ruling_spacing_coeff_cubic = self.auto_axis_index(from_right=True)


@dataclasses.dataclass
class Grating(optics.component.CylindricalComponent[SurfaceT]):
    name: Name = dataclasses.field(default_factory=lambda: Name('grating'))
    serial_number: np.ndarray = dataclasses.field(default_factory=lambda: np.array(''))
    manufacturing_number: np.ndarray = dataclasses.field(default_factory=lambda: np.array(''))
    inclination: u.Quantity = 0 * u.deg
    inclination_error: u.Quantity = 0 * u.deg
    roll: u.Quantity = 0 * u.deg
    roll_error: u.Quantity = 0 * u.deg
    twist: u.Quantity = 0 * u.deg
    twist_error: u.Quantity = 0 * u.deg
    tangential_radius: u.Quantity = np.inf * u.mm
    tangential_radius_error: u.Quantity = 0 * u.mm
    sagittal_radius: u.Quantity = np.inf * u.mm
    sagittal_radius_error: u.Quantity = 0 * u.mm
    mtf_degradation_factor: u.Quantity = 0 * u.dimensionless_unscaled
    slope_error: optics.surface.sag.SlopeErrorRMS = dataclasses.field(default_factory=optics.surface.sag.SlopeErrorRMS)
    ripple: optics.surface.sag.RippleRMS = dataclasses.field(default_factory=optics.surface.sag.RippleRMS)
    microroughness: optics.surface.sag.RoughnessRMS = dataclasses.field(default_factory=optics.surface.sag.RoughnessRMS)
    nominal_input_angle: u.Quantity = 0 * u.deg
    nominal_output_angle: u.Quantity = 0 * u.deg
    diffraction_order: u.Quantity = 0 << u.dimensionless_unscaled
    ruling_density: u.Quantity = 0 / u.mm
    ruling_density_error: u.Quantity = 0 / u.mm
    ruling_spacing_coeff_linear: u.Quantity = 0 * u.dimensionless_unscaled
    ruling_spacing_coeff_linear_error: u.Quantity = 0 * u.dimensionless_unscaled
    ruling_spacing_coeff_quadratic: u.Quantity = 0 / u.mm
    ruling_spacing_coeff_quadratic_error: u.Quantity = 0 / u.mm
    ruling_spacing_coeff_cubic: u.Quantity = 0 / u.mm ** 2
    ruling_spacing_coeff_cubic_error: u.Quantity = 0 / u.mm ** 2
    aper_wedge_angle: u.Quantity = 0 * u.deg
    inner_half_width: u.Quantity = 0 * u.mm
    outer_half_width: u.Quantity = 0 * u.mm
    border_width: u.Quantity = 0 * u.mm
    inner_border_width: u.Quantity = 0 * u.mm
    dynamic_clearance: u.Quantity = 0 * u.mm
    material: optics.surface.material.MultilayerMirror = dataclasses.field(
        default_factory=optics.surface.material.MultilayerMirror)
    witness: optics.surface.material.MultilayerMirror = dataclasses.field(
        default_factory=optics.surface.material.MultilayerMirror)

    @property
    def dataframe(self) -> pandas.DataFrame:
        dataframe = super().dataframe
        dataframe['inclination'] = [format.quantity(self.inclination.to(u.deg))]
        dataframe['tangential radius'] = [format.quantity(self.tangential_radius.to(u.mm))]
        dataframe['sagittal radius'] = [format.quantity(self.sagittal_radius.to(u.mm))]
        dataframe['slope error'] = [format.quantity(self.slope_error.value)]
        dataframe['ripple'] = [format.quantity(self.ripple.value)]
        dataframe['microroughness'] = [format.quantity(self.microroughness.value)]
        dataframe['nominal alpha'] = [format.quantity(self.nominal_input_angle.to(u.deg))]
        dataframe['nominal beta'] = [format.quantity(self.nominal_output_angle.to(u.deg))]
        dataframe['diffraction order'] = [format.quantity(self.diffraction_order)]
        dataframe['nominal ruling density'] = [format.quantity(self.ruling_density.to(1 / u.mm))]
        dataframe['linear ruling coefficient'] = [
            format.quantity(self.ruling_spacing_coeff_linear, scientific_notation=True)]
        dataframe['quadratic ruling coefficient'] = [
            format.quantity(self.ruling_spacing_coeff_quadratic, scientific_notation=True)]
        dataframe['cubic ruling coefficient'] = [
            format.quantity(self.ruling_spacing_coeff_cubic, scientific_notation=True)]
        dataframe['aperture wedge angle'] = [format.quantity(self.aper_wedge_angle.to(u.deg))]
        dataframe['inner half-width'] = [format.quantity(self.inner_half_width.to(u.mm))]
        dataframe['outer half-width'] = [format.quantity(self.outer_half_width.to(u.mm))]
        dataframe['border width'] = [format.quantity(self.border_width.to(u.mm))]
        dataframe['inner border width'] = [format.quantity(self.inner_border_width.to(u.mm))]
        dataframe['dynamic clearance'] = [format.quantity(self.dynamic_clearance.to(u.mm))]
        dataframe['substrate thickness'] = [format.quantity(self.substrate_thickness.to(u.mm))]
        return dataframe

    @property
    def is_toroidal(self) -> bool:
        return self.tangential_radius != self.sagittal_radius

    @property
    def is_vls(self) -> bool:
        a = self.ruling_spacing_coeff_linear != 0
        b = self.ruling_spacing_coeff_quadratic != 0
        c = self.ruling_spacing_coeff_cubic != 0
        return a or b or c

    @property
    def aper_wedge_half_angle(self) -> u.Quantity:
        return self.aper_wedge_angle / 2

    @property
    def height(self) -> u.Quantity:
        return self.outer_half_width + self.inner_half_width

    @property
    def height_mech(self):
        return self.height + self.border_width + self.inner_border_width

    @property
    def width_short(self) -> u.Quantity:
        return 2 * np.tan(self.aper_wedge_half_angle) * (-self.surface.aperture.apex_offset - self.inner_half_width)

    @property
    def width_long(self) -> u.Quantity:
        return 2 * np.tan(self.aper_wedge_half_angle) * (-self.surface.aperture.apex_offset + self.outer_half_width)

    @property
    def width_mech_short(self) -> u.Quantity:
        pos_x = -self.surface.aperture_mechanical.apex_offset - self.inner_half_width - self.inner_border_width
        return 2 * np.tan(self.aper_wedge_half_angle) * pos_x

    @property
    def width_mech_long(self) -> u.Quantity:
        pos_x = -self.surface.aperture_mechanical.apex_offset + self.outer_half_width + self.border_width
        return 2 * np.tan(self.aper_wedge_half_angle) * pos_x

    @property
    def dynamic_clearance_x(self):
        return self.dynamic_clearance / np.sin(self.aper_wedge_half_angle)

    def diffraction_angle(self, wavelength: u.Quantity, input_angle: u.Quantity = 0 * u.deg):
        return self.surface.rulings.diffraction_angle(wavelength=wavelength, input_angle=input_angle)

    @property
    def magnification_anamorphic(self) -> u.Quantity:
        return np.cos(self.nominal_input_angle) / np.cos(self.nominal_output_angle)

    @property
    def transform(self) -> transform.rigid.TransformList:
        return super().transform + transform.rigid.TransformList([
            transform.rigid.TiltY(self.inclination + self.inclination_error),
            transform.rigid.TiltX(self.twist + self.twist_error),
            transform.rigid.TiltZ(self.roll + self.roll_error),
        ])

    @property
    def surface(self) -> SurfaceT:
        surface = super().surface  # type: SurfaceT
        surface.is_stop = True
        surface.sag = optics.surface.sag.Toroidal(
            radius=self.sagittal_radius + self.sagittal_radius_error,
            radius_of_rotation=self.tangential_radius + self.tangential_radius_error,
        )
        surface.rulings = optics.surface.rulings.CubicPolySpacing(
            diffraction_order=self.diffraction_order,
            ruling_density=self.ruling_density + self.ruling_density_error,
            ruling_spacing_linear=self.ruling_spacing_coeff_linear + self.ruling_spacing_coeff_linear_error,
            ruling_spacing_quadratic=self.ruling_spacing_coeff_quadratic + self.ruling_spacing_coeff_quadratic_error,
            ruling_spacing_cubic=self.ruling_spacing_coeff_cubic + self.ruling_spacing_coeff_cubic_error,
        )
        surface.material = self.material
        side_border_x = self.border_width / np.sin(self.aper_wedge_half_angle) + self.dynamic_clearance_x
        surface.aperture = optics.surface.aperture.IsoscelesTrapezoid(
            apex_offset=-(self.cylindrical_radius - side_border_x),
            half_width_left=self.inner_half_width,
            half_width_right=self.outer_half_width,
            wedge_half_angle=self.aper_wedge_half_angle,
        )
        surface.aperture_mechanical = optics.surface.aperture.IsoscelesTrapezoid(
            apex_offset=-(self.cylindrical_radius - self.dynamic_clearance_x),
            half_width_left=self.inner_half_width + self.inner_border_width,
            half_width_right=self.outer_half_width + self.border_width,
            wedge_half_angle=self.aper_wedge_half_angle,
        )
        return surface

    def apply_gregorian_layout(
            self,
            magnification: u.Quantity,
            primary_focal_length: u.Quantity,
            primary_clear_radius: u.Quantity,
            back_focal_length: u.Quantity,
            detector_cylindrical_radius: u.Quantity,
            obscuration_margin: u.Quantity,
    ) -> 'Grating':
        """
        Computes an optimal placement for the grating based on the magnification, the primary location and the detector
        location.
        Based off of the procedure developed by Charles Kankelborg in SPIDER/spider/optics/design/configurations.ipynb
        :param magnification: Requested magnification of the Gregorian telescope
        :param primary_focal_length: Focal length of the parabolic primary mirror.
        :param primary_clear_radius: Radius of the usable area on the primary mirror
        :param back_focal_length: Distance from apex of primary to center of detector projected on the z-axis.
        :param detector_cylindrical_radius: Radial distance from the center of the detector to the axis of symmetry
        :param obscuration_margin: Size of the unusable border around the outside of the grating.
        :return: A new :py:class:esis.optics.components.Grating instance.
        """
        M = magnification
        f = primary_focal_length.to(u.mm).value
        D_p = 2 * primary_clear_radius.to(u.mm).value * np.cos(self.aper_wedge_half_angle)
        r_d = detector_cylindrical_radius.to(u.mm).value
        x_d = -back_focal_length.to(u.mm).value
        m_g = obscuration_margin.to(u.mm).value

        def gregorian_system(prams: typ.Tuple[float, ...]) -> typ.Tuple[float, ...]:
            x_g, r_g, D_g, h_g = prams
            eq1 = D_p * (x_g - f) - D_g * f
            eq2 = (D_p - D_g - 2 * m_g) * (x_g - f) - 2 * f * h_g
            eq3 = 2 * r_g - D_g + h_g
            eq4 = M ** 2 * ((x_g - f) ** 2 + r_g ** 2) - ((x_g - x_d) ** 2 + (r_d - r_g) ** 2)
            return eq1, eq2, eq3, eq4

        x_g = 1.2 * f
        r_g = D_p / 6
        D_g = D_p * (x_g - f) / f
        h_g = D_g / 4
        x_g, r_g, D_g, h_g = scipy.optimize.fsolve(gregorian_system, (x_g, r_g, D_g, h_g))
        x_g, r_g, D_g, h_g = x_g << u.mm, r_g << u.mm, D_g << u.mm, h_g << u.mm

        other = self.copy()
        other.piston = x_g
        other.cylindrical_radius = r_g
        other.inner_half_width = h_g / 2
        other.outer_half_width = h_g / 2
        # other.aper_cylindrical_radius = -r_g
        # other.inner_clear_radius = (D_g / 2 - h_g) / np.cos(self.aper_half_angle)
        # other.outer_clear_radius = (D_g / 2) / np.cos(self.aper_half_angle)

        return other

    def apply_poletto_prescription(
            self,
            wavelength_1: u.Quantity,
            wavelength_2: u.Quantity,
            magnification: u.Quantity,
            primary_focal_length: u.Quantity,
            detector_cylindrical_radius: u.Quantity,
            is_toroidal: bool = False,
            is_vls: bool = False,
    ) -> 'Grating':

        other = self.copy()

        m = other.diffraction_order

        lambda_1 = wavelength_1
        lambda_2 = wavelength_2
        lambda_c = (lambda_1 + lambda_2) / 2

        M_c = magnification

        piston_source = other.piston - primary_focal_length
        entrance_arm_length = np.sqrt(np.square(other.cylindrical_radius) + np.square(piston_source))
        entrance_arm_angle = -np.arctan2(other.cylindrical_radius, piston_source)

        exit_arm_length = magnification * entrance_arm_length
        exit_arm_radius = detector_cylindrical_radius - other.cylindrical_radius
        exit_arm_piston = np.sqrt(np.square(exit_arm_length) - np.square(exit_arm_radius))
        exit_arm_length = np.sqrt(np.square(exit_arm_radius) + np.square(exit_arm_piston))
        exit_arm_angle = np.arctan2(exit_arm_radius, exit_arm_piston)
        r_A = entrance_arm_length

        # Equation 17
        inclination_y = (M_c * np.sin(entrance_arm_angle) - np.sin(exit_arm_angle))
        inclination_x = M_c * np.cos(entrance_arm_angle) - np.cos(exit_arm_angle)
        other.inclination = np.arctan2(inclination_y, inclination_x)

        alpha = entrance_arm_angle - other.inclination
        beta_c = exit_arm_angle - other.inclination

        other.nominal_input_angle = alpha
        other.nominal_output_angle = beta_c

        cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
        cos2_alpha, sin2_alpha = np.square(cos_alpha), np.square(sin_alpha)

        cos_beta_c, sin_beta_c = np.cos(beta_c), np.sin(beta_c)
        cos2_beta_c, sin2_beta_c = np.square(cos_beta_c), np.square(sin_beta_c)

        # Grating equation (13)
        other.ruling_density = (np.sin(alpha) + np.sin(beta_c)) / (m * lambda_c)
        other.ruling_density_coeff_linear = 0 / u.mm ** 2
        other.ruling_density_coeff_quadratic = 0 / u.mm ** 3
        other.ruling_density_coeff_cubic = 0 / u.mm ** 4

        # Grating equation (13)
        beta_1 = other.surface.rulings.diffraction_angle(lambda_1, alpha)
        beta_2 = other.surface.rulings.diffraction_angle(lambda_2, alpha)

        cos_beta_1, cos_beta_2 = np.cos(beta_1), np.cos(beta_2)
        cos2_beta_1, cos2_beta_2 = np.square(cos_beta_1), np.square(cos_beta_2)

        calpha_plus_cbeta1 = cos_alpha + cos_beta_1
        calpha_plus_cbeta2 = cos_alpha + cos_beta_2

        if not is_toroidal:

            # Spherical uniform line spacing (SULS)
            if not is_vls:
                raise NotImplementedError

            # Spherical variable line spacing (SVLS)
            else:

                # Equation 31
                rho = R = r_A * (cos_alpha + cos_beta_c) * M_c / (1 + M_c)

                # Equation 26
                other.ruling_density_coeff_linear = (M_c + 1) * sin2_alpha / (m * lambda_c * r_A)

        else:

            # Toroidal uniform line spacing (TULS)
            if not is_vls:

                # Equation 24
                R = r_A * (cos_alpha + cos_beta_c) * M_c / (1 + M_c)

                # Equation 25
                rho = cos2_beta_1 / (1 / R - (cos_alpha + cos_beta_1) / r_A)

            # Toroidal variable line spacing (TVLS)
            else:

                # Equation 35
                rho = r_A * (cos_alpha + cos_beta_c) * M_c / (1 + M_c)

                # Equation 38
                K_1 = cos2_alpha / r_A + cos2_beta_1 * (calpha_plus_cbeta1 / rho - 1 / r_A)

                # Equation 39
                K_2 = cos2_alpha / r_A + cos2_beta_2 * (calpha_plus_cbeta2 / rho - 1 / r_A)

                # Equation 36
                R = (lambda_1 * calpha_plus_cbeta2 - lambda_2 * calpha_plus_cbeta1) / (lambda_1 * K_2 - lambda_2 * K_1)

                # Equation 37
                sigma_1_denominator = lambda_1 * calpha_plus_cbeta2 - lambda_2 * calpha_plus_cbeta1
                sigma_1 = (1 / m) * (K_2 * calpha_plus_cbeta1 - K_1 * calpha_plus_cbeta2) / sigma_1_denominator
                other.ruling_density_coeff_linear = sigma_1

                # Equation 33
                r_Bh_c = poletto.spectral_focal_curve(lambda_c, m, alpha, beta_c, r_A, R, sigma_1)

                c_alpha = cos_alpha / r_A - 1 / R
                c_beta_c = cos_beta_c / r_Bh_c - 1 / R
                sigma_2_1 = sin_alpha * cos_alpha / r_A * c_alpha
                sigma_2_2 = sin_beta_c * cos_beta_c / r_Bh_c * c_beta_c
                other.ruling_density_coeff_quadratic = - (3 / (2 * m * lambda_c)) * (sigma_2_1 + sigma_2_2)

                sigma_3_1 = 4 * sin2_alpha * cos_alpha / np.square(r_A) * c_alpha
                sigma_3_2 = -cos2_alpha / r_A * np.square(c_alpha)
                sigma_3_3 = 1 / np.square(R) * (1 / r_A - cos_alpha / R)
                sigma_3_4 = 4 * sin2_beta_c * cos_beta_c / np.square(r_Bh_c) * c_beta_c
                sigma_3_5 = -cos2_beta_c / r_Bh_c * np.square(c_beta_c)
                sigma_3_6 = 1 / np.square(R) * (1 / r_Bh_c - cos_beta_c / R)
                sigma_3 = -1 / (2 * m * lambda_c)
                sigma_3 = sigma_3 * (sigma_3_1 + sigma_3_2 + sigma_3_3 + sigma_3_4 + sigma_3_5 + sigma_3_6)
                other.ruling_density_coeff_cubic = sigma_3

        other.sagittal_radius = rho
        other.tangential_radius = R

        return other
