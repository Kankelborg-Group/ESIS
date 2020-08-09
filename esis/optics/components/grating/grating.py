import typing as typ
import dataclasses
import numpy as np
import scipy.optimize
import astropy.units as u
from kgpy import Name, optics
from ... import poletto
from .. import Component

__all__ = ['Grating']

AperSurfT = optics.surface.Toroidal[None, optics.aperture.IsoscelesTrapezoid]
MainSurfT = optics.surface.ToroidalVariableLineSpaceGrating[optics.material.Mirror, optics.aperture.IsoscelesTrapezoid]

default_name = Name('grating')


@dataclasses.dataclass
class Grating(Component):
    name: Name = dataclasses.field(default_factory=lambda: default_name)
    tangential_radius: u.Quantity = np.inf * u.mm
    sagittal_radius: u.Quantity = np.inf * u.mm
    nominal_input_angle: u.Quantity = 0 * u.deg
    nominal_output_angle: u.Quantity = 0 * u.deg
    diffraction_order: u.Quantity = 0 << u.dimensionless_unscaled
    groove_density: u.Quantity = 0 / u.mm
    groove_density_coeff_linear: u.Quantity = 0 / (u.mm ** 2)
    groove_density_coeff_quadratic: u.Quantity = 0 / (u.mm ** 3)
    groove_density_coeff_cubic: u.Quantity = 0 / (u.mm ** 4)
    piston: u.Quantity = 0 * u.mm
    channel_radius: u.Quantity = 0 * u.mm
    channel_angle: u.Quantity = 0 * u.deg
    inclination: u.Quantity = 0 * u.deg
    aper_half_angle: u.Quantity = 0 * u.deg
    aper_decenter_x: u.Quantity = 0 * u.mm
    inner_clear_radius: u.Quantity = 0 * u.mm
    outer_clear_radius: u.Quantity = 0 * u.mm
    inner_border_width: u.Quantity = 0 * u.mm
    outer_border_width: u.Quantity = 0 * u.mm
    side_border_width: u.Quantity = 0 * u.mm
    dynamic_clearance: u.Quantity = 0 * u.mm
    substrate_thickness: u.Quantity = 0 * u.mm

    @property
    def is_toroidal(self) -> bool:
        return self.tangential_radius != self.sagittal_radius

    @property
    def is_vls(self) -> bool:
        a = self.groove_density_coeff_linear != 0
        b = self.groove_density_coeff_quadratic != 0
        c = self.groove_density_coeff_cubic != 0
        return a or b or c

    @property
    def dynamic_clearance_x(self):
        return self.dynamic_clearance / np.sin(self.aper_half_angle)

    def diffraction_angle(self, wavelength: u.Quantity, input_angle: u.Quantity = 0 * u.deg):
        return self.main_surface.diffraction_angle(wavelength=wavelength, input_angle=input_angle)

    @property
    def surface(self) -> AperSurfT:
        side_border_x = self.side_border_width / np.sin(self.aper_half_angle) + self.dynamic_clearance_x
        return optics.surface.Toroidal(
            name=Name('aper'),
            radius=self.sagittal_radius,
            aperture=optics.aperture.IsoscelesTrapezoid(
                decenter=optics.coordinate.Decenter(x=self.aper_decenter_x + side_border_x),
                inner_radius=self.inner_clear_radius - side_border_x,
                outer_radius=self.outer_clear_radius - side_border_x,
                wedge_half_angle=self.aper_half_angle,
            ),
            radius_of_rotation=self.tangential_radius,
        )

    @property
    def main_surface(self):
        return optics.surface.ToroidalVariableLineSpaceGrating(
            name=Name('main'),
            radius=self.sagittal_radius,
            material=optics.material.Mirror(thickness=-self.substrate_thickness),
            aperture=optics.aperture.IsoscelesTrapezoid(
                is_active=False,
                decenter=optics.coordinate.Decenter(x=self.aper_decenter_x + self.dynamic_clearance_x),
                inner_radius=self.inner_clear_radius - self.inner_border_width - self.dynamic_clearance_x,
                outer_radius=self.outer_clear_radius + self.outer_border_width - self.dynamic_clearance_x,
                wedge_half_angle=self.aper_half_angle,
            ),
            radius_of_rotation=self.tangential_radius,
            diffraction_order=self.diffraction_order,
            groove_density=self.groove_density,
            coeff_linear=self.groove_density_coeff_linear,
            coeff_quadratic=self.groove_density_coeff_quadratic,
            coeff_cubic=self.groove_density_coeff_cubic,
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
                    tilt=optics.coordinate.Tilt(y=self.inclination)
                ),
            ],
        )

    def copy(self) -> 'Grating':
        return Grating(
            name=self.name.copy(),
            tangential_radius=self.tangential_radius.copy(),
            sagittal_radius=self.sagittal_radius.copy(),
            groove_density=self.groove_density.copy(),
            piston=self.piston.copy(),
            channel_radius=self.channel_radius.copy(),
            channel_angle=self.channel_angle.copy(),
            inclination=self.inclination.copy(),
            aper_half_angle=self.aper_half_angle.copy(),
            aper_decenter_x=self.aper_decenter_x.copy(),
            inner_clear_radius=self.inner_clear_radius.copy(),
            outer_clear_radius=self.outer_clear_radius.copy(),
            inner_border_width=self.inner_border_width.copy(),
            outer_border_width=self.outer_border_width.copy(),
            side_border_width=self.side_border_width.copy(),
            substrate_thickness=self.substrate_thickness.copy(),
            groove_density_coeff_linear=self.groove_density_coeff_linear.copy(),
            groove_density_coeff_quadratic=self.groove_density_coeff_quadratic.copy(),
            groove_density_coeff_cubic=self.groove_density_coeff_cubic.copy(),
        )

    def apply_gregorian_layout(
            self,
            magnification: u.Quantity,
            primary_focal_length: u.Quantity,
            primary_clear_radius: u.Quantity,
            back_focal_length: u.Quantity,
            detector_channel_radius: u.Quantity,
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
        :param detector_channel_radius: Radial distance from the center of the detector to the axis of symmetry
        :param obscuration_margin: Size of the unusable border around the outside of the grating.
        :return: A new :py:class:esis.optics.components.Grating instance.
        """
        M = magnification
        f = primary_focal_length.to(u.mm).value
        D_p = 2 * primary_clear_radius.to(u.mm).value
        r_d = detector_channel_radius.to(u.mm).value
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
        other.channel_radius = r_g
        other.aper_decenter_x = -r_g
        other.inner_clear_radius = D_g / 2 - h_g
        other.outer_clear_radius = D_g / 2

        return other

    def apply_poletto_prescription(
            self,
            wavelength_1: u.Quantity,
            wavelength_2: u.Quantity,
            magnification: u.Quantity,
            primary_focal_length: u.Quantity,
            detector_channel_radius: u.Quantity,
            is_toroidal: bool = False,
            is_vls: bool = False,
    ) -> 'Grating':

        other = self.copy()

        m = 1 * u.dimensionless_unscaled
        other.diffraction_order = m

        lambda_1 = wavelength_1
        lambda_2 = wavelength_2
        lambda_c = (lambda_1 + lambda_2) / 2

        M_c = magnification

        entrance_arm_length = np.sqrt(np.square(other.channel_radius) + np.square(other.piston - primary_focal_length))
        entrance_arm_angle = -np.arctan2(other.channel_radius, other.piston - primary_focal_length)

        exit_arm_length = magnification * entrance_arm_length
        exit_arm_radius = detector_channel_radius - other.channel_radius
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
        other.groove_density = (np.sin(alpha) + np.sin(beta_c)) / (m * lambda_c)
        other.groove_density_coeff_linear = 0 / u.mm ** 2
        other.groove_density_coeff_quadratic = 0 / u.mm ** 3
        other.groove_density_coeff_cubic = 0 / u.mm ** 4

        # Grating equation (13)
        beta_1 = other.main_surface.diffraction_angle(lambda_1, alpha)
        beta_2 = other.main_surface.diffraction_angle(lambda_2, alpha)

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
                other.groove_density_coeff_linear = (M_c + 1) * sin2_alpha / (m * lambda_c * r_A)

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
                other.groove_density_coeff_linear = sigma_1

                # Equation 33
                r_Bh_c = poletto.spectral_focal_curve(lambda_c, m, alpha, beta_c, r_A, R, sigma_1)

                c_alpha = cos_alpha / r_A - 1 / R
                c_beta_c = cos_beta_c / r_Bh_c - 1 / R
                sigma_2_1 = sin_alpha * cos_alpha / r_A * c_alpha
                sigma_2_2 = sin_beta_c * cos_beta_c / r_Bh_c * c_beta_c
                other.groove_density_coeff_quadratic = - (3 / (2 * m * lambda_c)) * (sigma_2_1 + sigma_2_2)

                sigma_3_1 = 4 * sin2_alpha * cos_alpha / np.square(r_A) * c_alpha
                sigma_3_2 = -cos2_alpha / r_A * np.square(c_alpha)
                sigma_3_3 = 1 / np.square(R) * (1 / r_A - cos_alpha / R)
                sigma_3_4 = 4 * sin2_beta_c * cos_beta_c / np.square(r_Bh_c) * c_beta_c
                sigma_3_5 = -cos2_beta_c / r_Bh_c * np.square(c_beta_c)
                sigma_3_6 = 1 / np.square(R) * (1 / r_Bh_c - cos_beta_c / R)
                sigma_3 = -1 / (2 * m * lambda_c)
                sigma_3 = sigma_3 * (sigma_3_1 + sigma_3_2 + sigma_3_3 + sigma_3_4 + sigma_3_5 + sigma_3_6)
                other.groove_density_coeff_cubic = sigma_3

        other.sagittal_radius = rho
        other.tangential_radius = R

        return other
