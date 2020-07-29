import typing as typ
import dataclasses
import numpy as np
import astropy.units as u
from kgpy import Name, optics, vector
from . import components as cmps

__all__ = ['Optics']

default_name = Name('SPIDER')


@dataclasses.dataclass
class Optics:
    name: Name = dataclasses.field(default_factory=lambda: default_name)
    components: cmps.Components = dataclasses.field(default_factory=lambda: cmps.Components())
    wavelengths: u.Quantity = 0 * u.nm
    field_limit: u.Quantity = 0 * u.deg
    pupil_samples: int = 10
    field_samples: int = 10

    def __post_init__(self):
        self.update()

    def update(self) -> typ.NoReturn:
        self._system = None

    @property
    def system(self) -> optics.System:
        if self._system is None:
            self._system = self._calc_system()
        return self._system

    def _calc_system(self) -> optics.System:
        return optics.System(
            object_surface=optics.surface.ObjectSurface(thickness=np.inf * u.mm),
            surfaces=self.components,
            stop_surface=self.components.grating.surface,
            wavelengths=self.wavelengths,
            pupil_samples=self.pupil_samples,
            field_min=-self.field_limit,
            field_max=self.field_limit,
            field_samples=self.field_samples,
        )

    @classmethod
    def esis_as_designed(
            cls,
            pupil_samples: int = 10,
            field_samples: int = 10,
    ) -> 'Optics':
        num_sides = 8
        num_channels = 4
        deg_per_channel = 360 * u.deg / num_sides
        channel_offset_angle = deg_per_channel / 2
        channel_angle = np.linspace(0 * u.deg, num_channels * deg_per_channel, num_channels, endpoint=False)
        # channel_angle += channel_offset_angle
        channel_angle = 0 * u.deg

        tuffet_x1, tuffet_y1 = 2.54 * u.mm, 37.1707 * u.mm
        tuffet_x2, tuffet_y2 = 24.4876 * u.mm, 28.0797 * u.mm
        tuffet_dx, tuffet_dy = tuffet_x2 - tuffet_x1, tuffet_y2 - tuffet_y1
        tuffet_slope = tuffet_dy / tuffet_dx
        tuffet_radius = tuffet_y1 - tuffet_slope * tuffet_x1

        primary_piston = 1000 * u.mm
        primary_clear_radius = 77.9 * u.mm

        grating_radius = 597.830 * u.mm
        grating_piston = -374.7 * u.mm
        grating_channel_radius = 2.074999998438000e1 * u.mm
        grating_border_width = 2 * u.mm
        grating_lower_border_width = 4.86 * u.mm
        grating_inner_clear_radius = grating_channel_radius - (13.02 * u.mm - grating_lower_border_width)
        groove_density = 2.586608603456000 / u.um
        d0 = 1 / groove_density
        d_c1 = -3.3849e-5 * (u.um / u.mm)
        d_c2 = -1.3625e-7 * (u.um / u.mm ** 2)

        grating_to_filter_distance = 1.301661998854058 * u.m
        filter_piston = grating_to_filter_distance + grating_piston

        pix_half_width = 15 * u.um / 2

        field_limit = 0.09561 * u.deg
        return cls(
            name=Name('ESIS'),
            components=cmps.Components(
                front_aperture=cmps.FrontAperture(
                    piston=-500 * u.mm,
                    clear_radius=100 * u.mm,
                ),
                central_obscuration=cmps.CentralObscuration(
                    piston=primary_piston - 1404.270 * u.mm,
                    obscured_radius=tuffet_radius,
                    num_sides=num_sides,
                ),
                primary=cmps.Primary(
                    radius=-2 * primary_piston,
                    piston=primary_piston,
                    num_sides=num_sides,
                    clear_radius=primary_clear_radius,
                    border_width=83.7 * u.mm - primary_clear_radius,
                    substrate_thickness=30 * u.mm,
                ),
                field_stop=cmps.FieldStop(
                    clear_radius=1.82 * u.mm,
                    mech_radius=2.81 * u.mm,
                    num_sides=num_sides,
                ),
                grating=cmps.Grating(
                    tangential_radius=grating_radius,
                    sagittal_radius=grating_radius,
                    groove_density=groove_density,
                    piston=grating_piston,
                    channel_radius=grating_channel_radius,
                    channel_angle=channel_angle,
                    inclination=-4.469567242792327 * u.deg,
                    aper_half_angle=deg_per_channel / 2,
                    aper_decenter_x=-grating_channel_radius,
                    groove_density_coeff_linear=-d_c1 / np.square(d0),
                    groove_density_coeff_quadratic=(np.square(d_c1) - d0 * d_c2) / np.power(d0, 3),
                    # groove_density_coeff_cubic=(np.power(d_c1, 3) - 2 * d0 * d_c1 * d_c2) / np.power(d0, 4),
                    # groove_density_coeff_linear=d_c1,
                    # groove_density_coeff_quadratic=d_c2,
                    groove_density_coeff_cubic=0 / u.mm ** 4,
                    inner_clear_radius=grating_inner_clear_radius,
                    outer_clear_radius=grating_channel_radius + (10.49 * u.mm - grating_border_width),
                    inner_border_width=grating_lower_border_width,
                    outer_border_width=grating_border_width,
                    side_border_width=grating_border_width,
                    dynamic_clearance=1.25 * u.mm,
                    substrate_thickness=10 * u.mm,
                ),
                filter=cmps.Filter(
                    piston=filter_piston,
                    channel_radius=95.9 * u.mm,
                    channel_angle=channel_angle,
                    inclination=-3.45 * u.deg,
                    clear_radius=15.9 * u.mm,
                ),
                detector=cmps.Detector(
                    piston=filter_piston + 200 * u.mm,
                    channel_radius=108 * u.mm,
                    channel_angle=channel_angle,
                    inclination=-12.252 * u.deg,
                    pix_half_width_x=pix_half_width,
                    pix_half_width_y=pix_half_width,
                    npix_x=2048,
                    npix_y=1024,
                ),
            ),
            wavelengths=[[629.7, 609.8, 584.3, ]] * u.AA,
            # wavelengths=[629.7, 609.8, 584.3, ] * u.AA,
            # wavelengths=[[629.7]] * u.AA,
            field_limit=vector.from_components(field_limit, field_limit).to(u.arcmin),
            pupil_samples=pupil_samples,
            field_samples=field_samples,
        )

    @classmethod
    def esis_from_poletto(
            cls,
            pupil_samples: int = 10,
            field_samples: int = 10,
    ):
        esis = cls.esis_as_designed(pupil_samples=pupil_samples, field_samples=field_samples)

        grating = esis.components.grating
        detector = esis.components.detector

        new_grating, new_detector = cls.poletto_tvls_grating_and_detector(
            wavelength_1=esis.wavelengths[..., 0],
            wavelength_2=esis.wavelengths[..., ~0],
            magnification=4,
            grating_channel_radius=grating.channel_radius,
            grating_piston=grating.piston,
            detector_channel_radius=detector.channel_radius,
        )

        grating.tangential_radius = new_grating.tangential_radius
        grating.sagittal_radius = new_grating.sagittal_radius
        grating.groove_density = new_grating.groove_density
        grating.inclination = new_grating.inclination
        grating.groove_density_coeff_linear = new_grating.groove_density_coeff_linear
        grating.groove_density_coeff_quadratic = new_grating.groove_density_coeff_quadratic
        grating.groove_density_coeff_cubic = new_grating.groove_density_coeff_cubic

        detector.piston = new_detector.piston
        detector.inclination = new_detector.inclination

        esis.update()

        return esis

    @classmethod
    def poletto_tvls_grating_and_detector(
            cls,
            wavelength_1: u.Quantity,
            wavelength_2: u.Quantity,
            magnification: float,
            grating_channel_radius: u.Quantity,
            grating_piston: u.Quantity,
            detector_channel_radius: u.Quantity,
    ) -> typ.Tuple[cmps.Grating, cmps.Detector]:
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

        grating = cmps.Grating(
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

        detector = cmps.Detector(
            piston=detector_piston,
            channel_radius=detector_channel_radius,
            inclination=detector_inclination,
        )

        return grating, detector
