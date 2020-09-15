import typing as typ
import numpy as np
import astropy.units as u
from kgpy import vector
# from . import Grating, Detector

__all__ = [
    # 'calc_grating_and_detector',
    'diffraction_angle_from_grating_equation',
    'spectral_focal_curve',
    'spatial_focal_curve',
]


# def calc_grating_and_detector(
#         wavelength_1: u.Quantity,
#         wavelength_2: u.Quantity,
#         source_piston: u.Quantity,
#         magnification: float,
#         grating_channel_radius: u.Quantity,
#         grating_piston: u.Quantity,
#         detector_channel_radius: u.Quantity,
#         diffraction_order: u.Quantity = 1 * u.dimensionless_unscaled,
#         use_toroidal_grating: bool = False,
#         use_vls_grating: bool = False,
#         use_one_wavelength_detector_tilt: bool = False,
# ) -> typ.Tuple[Grating, Detector]:
#     m = 1
#
#     lambda_1 = wavelength_1
#     lambda_2 = wavelength_2
#     lambda_c = (lambda_1 + lambda_2) / 2
#
#     M_c = magnification
#
#     entrance_arm_length = np.sqrt(np.square(grating_channel_radius) + np.square(grating_piston - source_piston))
#     entrance_arm_angle = -np.arctan2(grating_channel_radius, grating_piston - source_piston)
#
#     exit_arm_length = magnification * entrance_arm_length
#     exit_arm_radius = detector_channel_radius - grating_channel_radius
#     exit_arm_piston = np.sqrt(np.square(exit_arm_length) - np.square(exit_arm_radius))
#     exit_arm_length = np.sqrt(np.square(exit_arm_radius) + np.square(exit_arm_piston))
#     exit_arm_angle = np.arctan2(exit_arm_radius, exit_arm_piston)
#
#     # Equation 17
#     r_A = entrance_arm_length
#     r_B = exit_arm_length
#
#     # Equation 17
#     inclination_y = (M_c * np.sin(entrance_arm_angle) - np.sin(exit_arm_angle))
#     inclination_x = M_c * np.cos(entrance_arm_angle) - np.cos(exit_arm_angle)
#     grating_inclination = np.arctan2(inclination_y, inclination_x)
#
#     alpha = entrance_arm_angle - grating_inclination
#     beta_c = exit_arm_angle - grating_inclination
#
#     cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
#     cos2_alpha, sin2_alpha = np.square(cos_alpha), np.square(sin_alpha)
#
#     cos_beta_c, sin_beta_c = np.cos(beta_c), np.sin(beta_c)
#     cos2_beta_c, sin2_beta_c = np.square(cos_beta_c), np.square(sin_beta_c)
#
#     # Grating equation (13)
#     sigma_0 = (np.sin(alpha) + np.sin(beta_c)) / (m * lambda_c)
#     sigma_1 = 0 / (u.mm ** 2)
#     sigma_2: u.Quantity = 0 / (u.mm ** 3)
#     sigma_3: u.Quantity = 0 / (u.mm ** 4)
#
#     # Grating equation (13)
#     beta_1 = np.arcsin(m * lambda_1 * sigma_0 - sin_alpha)
#     beta_2 = np.arcsin(m * lambda_2 * sigma_0 - sin_alpha)
#
#     cos_beta_1, cos_beta_2 = np.cos(beta_1), np.cos(beta_2)
#     cos2_beta_1, cos2_beta_2 = np.square(cos_beta_1), np.square(cos_beta_2)
#
#     calpha_plus_cbeta1 = cos_alpha + cos_beta_1
#     calpha_plus_cbeta2 = cos_alpha + cos_beta_2
#
#     if not use_toroidal_grating:
#
#         # Spherical uniform line spacing (SULS)
#         if not use_vls_grating:
#             raise NotImplementedError
#
#         # Spherical variable line spacing (SVLS)
#         else:
#
#             # Equation 31
#             rho = R = r_A * (cos_alpha + cos_beta_c) * M_c / (1 + M_c)
#
#             # Equation 26
#             sigma_1 = (M_c + 1) * sin2_alpha / (m * lambda_c * r_A)
#
#     else:
#
#         # Toroidal uniform line spacing (TULS)
#         if not use_vls_grating:
#
#             # Equation 24
#             R = r_A * (cos_alpha + cos_beta_c) * M_c / (1 + M_c)
#
#             # Equation 25
#             rho = cos2_beta_1 / (1 / R - (cos_alpha + cos_beta_1) / r_A)
#
#         # Toroidal variable line spacing (TVLS)
#         else:
#
#             # Equation 35
#             rho = r_A * (cos_alpha + cos_beta_c) * M_c / (1 + M_c)
#
#             # Equation 38
#             K_1 = cos2_alpha / r_A + cos2_beta_1 * (calpha_plus_cbeta1 / rho - 1 / r_A)
#
#             # Equation 39
#             K_2 = cos2_alpha / r_A + cos2_beta_2 * (calpha_plus_cbeta2 / rho - 1 / r_A)
#
#             # Equation 36
#             R = (lambda_1 * calpha_plus_cbeta2 - lambda_2 * calpha_plus_cbeta1) / (lambda_1 * K_2 - lambda_2 * K_1)
#
#             # Equation 37
#             sigma_1_denominator = lambda_1 * calpha_plus_cbeta2 - lambda_2 * calpha_plus_cbeta1
#             sigma_1 = (1 / m) * (K_2 * calpha_plus_cbeta1 - K_1 * calpha_plus_cbeta2) / sigma_1_denominator
#
#             # Equation 33
#             r_Bh_c = spectral_focal_curve(lambda_c, m, alpha, beta_c, r_A, R, sigma_1)
#
#             c_alpha = cos_alpha / r_A - 1 / R
#             c_beta_c = cos_beta_c / r_Bh_c - 1 / R
#             sigma_2_1 = sin_alpha * cos_alpha / r_A * c_alpha
#             sigma_2_2 = sin_beta_c * cos_beta_c / r_Bh_c * c_beta_c
#             sigma_2 = - (3 / (2 * m * lambda_c)) * (sigma_2_1 + sigma_2_2)
#
#             sigma_3_1 = 4 * sin2_alpha * cos_alpha / np.square(r_A) * c_alpha
#             sigma_3_2 = -cos2_alpha / r_A * np.square(c_alpha)
#             sigma_3_3 = 1 / np.square(R) * (1 / r_A - cos_alpha / R)
#             sigma_3_4 = 4 * sin2_beta_c * cos_beta_c / np.square(r_Bh_c) * c_beta_c
#             sigma_3_5 = -cos2_beta_c / r_Bh_c * np.square(c_beta_c)
#             sigma_3_6 = 1 / np.square(R) * (1 / r_Bh_c - cos_beta_c / R)
#             sigma_3 = -1 / (2 * m * lambda_c) * (sigma_3_1 + sigma_3_2 + sigma_3_3 + sigma_3_4 + sigma_3_5 + sigma_3_6)
#
#     grating = Grating(
#         tangential_radius=R,
#         sagittal_radius=rho,
#         nominal_input_angle=alpha,
#         nominal_output_angle=beta_c,
#         diffraction_order=diffraction_order,
#         groove_density=sigma_0,
#         piston=grating_piston,
#         channel_radius=grating_channel_radius,
#         inclination=grating_inclination,
#         groove_density_coeff_linear=sigma_1,
#         groove_density_coeff_quadratic=sigma_2,
#         groove_density_coeff_cubic=sigma_3,
#     )
#
#     if not use_one_wavelength_detector_tilt:
#         detector = two_point_detector(lambda_1, lambda_2, r_A, grating)
#     else:
#         detector = one_point_detector(lambda_c, source_piston, magnification, r_A, grating,
#                                       use_toroidal_grating=use_toroidal_grating, use_vls_grating=use_vls_grating)
#     detector.channel_radius = detector_channel_radius
#
#     return grating, detector
#
#
# def one_point_detector(
#         wavelength: u.Quantity,
#         source_piston: u.Quantity,
#         magnification: float,
#         entrance_arm_radius: u.Quantity,
#         grating: Grating,
#         use_toroidal_grating: bool = False,
#         use_vls_grating: bool = False,
# ) -> Detector:
#
#     if not use_toroidal_grating and use_vls_grating:
#         f = source_piston
#         M = magnification
#         alpha = grating.nominal_input_angle
#         beta = grating.diffraction_angle(wavelength, alpha)
#         r_A = entrance_arm_radius
#         r_B = M * r_A
#         R = grating.tangential_radius
#         x_g = grating.piston
#         r_g = grating.channel_radius
#
#         sin_alpha, cos_alpha, tan_alpha = np.sin(alpha), np.cos(alpha), np.tan(alpha)
#         sin_beta, cos_beta, tan_beta = np.sin(beta), np.cos(beta), np.tan(beta)
#
#         tanphi_1spec = r_B * tan_beta / (R * cos_beta) - tan_beta
#         tanphi_1spat = r_B * sin_beta / R
#         tanphi_2spec = r_B * (tan_beta - tan_alpha) / (R * cos_beta) - M * r_g * cos_alpha / ((x_g - f) * cos_beta)
#         tanphi_2spat = r_B * (sin_beta - tan_alpha * cos_beta) / R - M * r_g * cos_beta / ((x_g - f) * cos_alpha)
#         phi = np.arctan([tanphi_1spec, tanphi_1spat, tanphi_2spec, tanphi_2spat]) << u.rad
#         phi_avg = (np.amax(phi) + np.amin(phi)) / 2  # compromise value for detector tilt
#
#         return Detector(
#             piston=grating.piston - r_B * np.cos(beta + grating.inclination),
#             inclination=-phi_avg,
#         )
#     else:
#         raise ValueError('Only SVLS supported')
#
#
# def two_point_detector(
#         wavelength_1: u.Quantity,
#         wavelength_2: u.Quantity,
#         entrance_arm_radius: u.Quantity,
#         grating: Grating,
# ) -> Detector:
#
#     lambda_1, lambda_2 = wavelength_1, wavelength_2
#     m = grating.diffraction_order
#     alpha = grating.nominal_input_angle
#     beta_1, beta_2 = grating.diffraction_angle(lambda_1, alpha), grating.diffraction_angle(lambda_2, alpha)
#     r_A = entrance_arm_radius
#     R = grating.tangential_radius
#     sigma_1 = grating.ruling_density_coeff_linear
#
#     r_Bh_1 = spectral_focal_curve(lambda_1, m, alpha, beta_1, r_A, R, sigma_1)
#     r_Bh_2 = spectral_focal_curve(lambda_2, m, alpha, beta_2, r_A, R, sigma_1)
#
#     r_Bv_1 = spatial_focal_curve(alpha, beta_1, r_A, grating.sagittal_radius)
#     r_Bv_2 = spatial_focal_curve(alpha, beta_2, r_A, grating.sagittal_radius)
#
#     exit_arm_angle_1 = beta_1 + grating.inclination
#     exit_arm_angle_2 = beta_2 + grating.inclination
#
#     bv_1 = vector.from_components_cylindrical(r_Bv_1, exit_arm_angle_1)
#     bv_2 = vector.from_components_cylindrical(r_Bv_2, exit_arm_angle_2)
#     bh_1 = vector.from_components_cylindrical(r_Bh_1, exit_arm_angle_1)
#     bh_2 = vector.from_components_cylindrical(r_Bh_2, exit_arm_angle_2)
#
#     bv_ave = (bv_2 + bv_1) / 2
#     bh_ave = (bh_2 + bh_1) / 2
#     b_ave = (bv_ave + bh_ave) / 2
#     detector_piston = -bv_ave[vector.x] + grating.piston
#
#     dbh = bh_2 - bh_1
#     # dbh = bv_2 - bv_1
#     detector_inclination = np.arctan2(dbh[vector.y], dbh[vector.x]) + 90 * u.deg
#
#     return Detector(
#         piston=detector_piston,
#         inclination=detector_inclination,
#     )


def diffraction_angle_from_grating_equation(
        incidence_angle: u.Quantity,
        wavelength: u.Quantity,
        diffraction_order: u.Quantity,
        groove_density: u.Quantity,
) -> u.Quantity:
    """
    Equation 13 of Poletto and Thomas rearranged to solve for diffraction angle.
    :param incidence_angle: Signed angle between the surface normal and the incident ray
    :param wavelength: Wavelength of the incident light rays
    :param diffraction_order: Quantum number of diffracted ray.
    :param groove_density: Number of grooves per unit length on the grating surface.
    :return: Signed angle of the diffracted light
    """
    return np.arcsin(diffraction_order * wavelength * groove_density - np.sin(incidence_angle)) << u.rad


def spectral_focal_curve(
        wavelength: u.Quantity,
        diffraction_order: u.Quantity,
        alpha: u.Quantity,
        beta: u.Quantity,
        entrance_arm_radius: u.Quantity,
        grating_tangential_radius: u.Quantity,
        grating_linear_groove_coefficient: u.Quantity = 0 / (u.mm ** 2),
):
    """
    Equation 33 of Thomas and Poletto (2003)
    :param wavelength: The wavelength at which to evaluate the spectral focal curve
    :param diffraction_order: diffraction order of the light leaving the grating
    :param alpha: Nominal incidence angle of the beam on the grating.
    :param beta: Nominal diffraction angle of the beam off the grating.
    :param entrance_arm_radius: Distance from the center of then grating to the center of the source.
    :param grating_tangential_radius: Tangential radius of the diffraction grating
    :param grating_linear_groove_coefficient: Linear coefficient for the polynomial variable line spacing
    :return: The length of the exit arm for optimal spectral focus
    """
    m = diffraction_order
    cos_alpha, cos_beta = np.cos(alpha), np.cos(beta)
    cos2_alpha, cos2_beta = np.square(cos_alpha), np.square(cos_beta)
    r_A = entrance_arm_radius
    R = grating_tangential_radius
    sigma_1 = grating_linear_groove_coefficient
    return cos2_beta / (-cos2_alpha / r_A + (cos_alpha + cos_beta) / R - m * wavelength * sigma_1)


def spatial_focal_curve(
        alpha: u.Quantity,
        beta: u.Quantity,
        entrance_arm_radius: u.Quantity,
        grating_sagittal_radius: u.Quantity,
):
    """
    Equation 34 of Thomas and Poletto (2003)
    :param alpha: Nominal incidence angle of the beam on the grating.
    :param beta: Nominal diffraction angle of the beam off the grating.
    :param entrance_arm_radius: Distance from the center of then grating to the center of the source.
    :param grating_sagittal_radius: Sagittal radius of the diffraction grating
    :return: The length of the exit arm for optimal spatial focus
    """
    cos_alpha, cos_beta = np.cos(alpha), np.cos(beta)
    r_A = entrance_arm_radius
    rho = grating_sagittal_radius
    return 1 / (-1 / r_A + (cos_alpha + cos_beta) / rho)
