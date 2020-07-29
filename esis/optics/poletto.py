import typing as typ
import numpy as np
import astropy.units as u
from kgpy import vector
from . import components, Optics

__all__ = ['tvls_grating_and_detector']


def tvls_grating_and_detector(
        wavelength_1: u.Quantity,
        wavelength_2: u.Quantity,
        magnification: float,
        grating_channel_radius: u.Quantity,
        grating_piston: u.Quantity,
        detector_channel_radius: u.Quantity,
) -> typ.Tuple[components.Grating, components.Detector]:
    m = 1

    lambda_1 = wavelength_1
    lambda_2 = wavelength_2
    lambda_c = (lambda_1 + lambda_2) / 2
    # lambda_c = lambda_1

    M_c = magnification

    entrance_arm_length = np.sqrt(np.square(grating_channel_radius) + np.square(grating_piston))
    entrance_arm_angle = np.arctan2(grating_channel_radius, grating_piston) - 180 * u.deg

    exit_arm_length = magnification * entrance_arm_length
    exit_arm_radius = detector_channel_radius - grating_channel_radius
    exit_arm_piston = np.sqrt(np.square(exit_arm_length) - np.square(exit_arm_radius))
    detector_piston = exit_arm_piston + grating_piston
    exit_arm_length = np.sqrt(np.square(exit_arm_radius) + np.square(exit_arm_piston))
    exit_arm_angle = np.arctan2(exit_arm_radius, exit_arm_piston)

    # Equation 17
    r_A = entrance_arm_length
    r_B = exit_arm_length

    # Equation 17
    inclination_y = (M_c * np.sin(entrance_arm_angle) - np.sin(exit_arm_angle))
    inclination_x = M_c * np.cos(entrance_arm_angle) - np.cos(exit_arm_angle)
    grating_inclination = np.arctan2(inclination_y, inclination_x)

    alpha = entrance_arm_angle - grating_inclination
    beta_c = exit_arm_angle - grating_inclination

    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
    cos2_alpha, sin2_alpha = np.square(cos_alpha), np.square(sin_alpha)

    cos_beta_c, sin_beta_c = np.cos(beta_c), np.sin(beta_c)
    cos2_beta_c, sin2_beta_c = np.square(cos_beta_c), np.square(sin_beta_c)

    # Grating equation (13)
    sigma_0 = (np.sin(alpha) + np.sin(beta_c)) / (m * lambda_c)

    # Equation 35
    rho = r_A * (cos_alpha + cos_beta_c) * M_c / (1 + M_c)

    # Grating equation (13)
    beta_1 = np.arcsin(m * lambda_1 * sigma_0 - sin_alpha)
    beta_2 = np.arcsin(m * lambda_2 * sigma_0 - sin_alpha)

    cos_beta_1, cos_beta_2 = np.cos(beta_1), np.cos(beta_2)
    cos2_beta_1, cos2_beta_2 = np.square(cos_beta_1), np.square(cos_beta_2)

    calpha_plus_cbeta1 = cos_alpha + cos_beta_1
    calpha_plus_cbeta2 = cos_alpha + cos_beta_2

    # Equation 38
    K_1 = cos2_alpha / r_A + cos2_beta_1 * (calpha_plus_cbeta1 / rho - 1 / r_A)

    # Equation 39
    K_2 = cos2_alpha / r_A + cos2_beta_2 * (calpha_plus_cbeta2 / rho - 1 / r_A)

    # Equation 36
    R = (lambda_1 * calpha_plus_cbeta2 - lambda_2 * calpha_plus_cbeta1) / (lambda_1 * K_2 - lambda_2 * K_1)

    # Equation 37
    sigma_1_denominator = lambda_1 * calpha_plus_cbeta2 - lambda_2 * calpha_plus_cbeta1
    sigma_1 = (1 / m) * (K_2 * calpha_plus_cbeta1 - K_1 * calpha_plus_cbeta2) / sigma_1_denominator

    # Equation 33
    r_Bh_c = cos2_beta_c / (-cos2_alpha / r_A + (cos_alpha + cos_beta_c) / R - m * lambda_c * sigma_1)
    r_Bh_1 = cos2_beta_1 / (-cos2_alpha / r_A + (cos_alpha + cos_beta_1) / R - m * lambda_1 * sigma_1)
    r_Bh_2 = cos2_beta_2 / (-cos2_alpha / r_A + (cos_alpha + cos_beta_2) / R - m * lambda_2 * sigma_1)

    # Equation 34
    r_Bv_c = 1 / (-1 / r_A + (cos_alpha + cos_beta_c) / rho)
    r_Bv_1 = 1 / (-1 / r_A + (cos_alpha + cos_beta_1) / rho)
    r_Bv_2 = 1 / (-1 / r_A + (cos_alpha + cos_beta_2) / rho)

    c_alpha = cos_alpha / r_A - 1 / R
    c_beta_c = cos_beta_c / r_Bh_c - 1 / R
    sigma_2_1 = sin_alpha * cos_alpha / r_A * c_alpha
    sigma_2_2 = sin_beta_c * cos_beta_c / r_Bh_c * c_beta_c
    sigma_2 = - (3 / (2 * m * lambda_c)) * (sigma_2_1 + sigma_2_2)

    sigma_3_1 = 4 * sin2_alpha * cos_alpha / np.square(r_A) * c_alpha
    sigma_3_2 = -cos2_alpha / r_A * np.square(c_alpha)
    sigma_3_3 = 1 / np.square(R) * (1 / r_A - cos_alpha / R)
    sigma_3_4 = 4 * sin2_beta_c * cos_beta_c / np.square(r_Bh_c) * c_beta_c
    sigma_3_5 = -cos2_beta_c / r_Bh_c * np.square(c_beta_c)
    sigma_3_6 = 1 / np.square(R) * (1 / r_Bh_c - cos_beta_c / R)
    sigma_3 = -1 / (2 * m * lambda_c) * (sigma_3_1 + sigma_3_2 + sigma_3_3 + sigma_3_4 + sigma_3_5 + sigma_3_6)

    grating = components.Grating(
        tangential_radius=R,
        sagittal_radius=rho,
        groove_density=sigma_0,
        piston=grating_piston,
        channel_radius=grating_channel_radius,
        inclination=grating_inclination,
        groove_density_coeff_linear=sigma_1,
        groove_density_coeff_quadratic=sigma_2,
        groove_density_coeff_cubic=sigma_3,
    )

    r_B_c = (r_Bh_c + r_Bv_c) / 2
    r_B_1 = (r_Bh_1 + r_Bv_1) / 2
    r_B_2 = (r_Bh_2 + r_Bv_2) / 2

    exit_arm_angle_1 = beta_1 + grating_inclination
    exit_arm_angle_2 = beta_2 + grating_inclination
    b_1 = vector.from_components_cylindrical(r_B_1, exit_arm_angle_1)
    b_2 = vector.from_components_cylindrical(r_B_2, exit_arm_angle_2)
    db = b_2 - b_1
    b_ave = (b_2 + b_1) / 2
    detector_piston = b_ave[vector.x] + grating_piston
    detector_inclination = np.arctan2(db[vector.y], db[vector.x]) + 90 * u.deg

    detector = components.Detector(
        piston=detector_piston,
        channel_radius=detector_channel_radius,
        inclination=detector_inclination,
    )

    return grating, detector
