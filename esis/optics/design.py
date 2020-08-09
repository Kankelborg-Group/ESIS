import numpy as np
import astropy.units as u
from astropy import units as u

from kgpy import Name, vector
from . import components, Optics, poletto

__all__ = ['final', 'final_from_poletto']


def final(
        pupil_samples: int = 10,
        field_samples: int = 10,
) -> Optics:
    """
    Final ESIS optical design prepared by Charles Kankelborg and Hans Courrier.
    :param pupil_samples: Number of rays per axis across the pupil.
    :param field_samples: Number of rays per axis across the field.
    :return: An instance of the as-designed ESIS optics model.
    """
    num_sides = 8
    num_channels = 4
    deg_per_channel = 360 * u.deg / num_sides
    channel_offset_angle = deg_per_channel / 2
    channel_angle = np.linspace(0 * u.deg, num_channels * deg_per_channel, num_channels, endpoint=False)
    # channel_angle += channel_offset_angle
    channel_angle = 180 * u.deg

    tuffet_x1, tuffet_y1 = 2.54 * u.mm, 37.1707 * u.mm
    tuffet_x2, tuffet_y2 = 24.4876 * u.mm, 28.0797 * u.mm
    tuffet_dx, tuffet_dy = tuffet_x2 - tuffet_x1, tuffet_y2 - tuffet_y1
    tuffet_slope = tuffet_dy / tuffet_dx
    tuffet_radius = tuffet_y1 - tuffet_slope * tuffet_x1

    primary_focal_length = 1000 * u.mm
    primary_clear_radius = 77.9 * u.mm

    grating_radius = 597.830 * u.mm
    grating_piston = primary_focal_length + 374.7 * u.mm
    grating_channel_radius = 2.074999998438000e1 * u.mm
    grating_border_width = 2 * u.mm
    grating_lower_border_width = 4.86 * u.mm
    grating_inner_clear_radius = grating_channel_radius - (13.02 * u.mm - grating_lower_border_width)
    groove_density = 2.586608603456000 / u.um
    d0 = 1 / groove_density
    d_c1 = -3.3849e-5 * (u.um / u.mm)
    d_c2 = -1.3625e-7 * (u.um / u.mm ** 2)

    grating_to_filter_distance = 1.301661998854058 * u.m
    filter_piston = grating_piston - grating_to_filter_distance

    pix_half_width = 15 * u.um / 2

    field_limit = 0.09561 * u.deg
    return Optics(
        name=Name('ESIS'),
        components=components.Components(
            front_aperture=components.FrontAperture(
                piston=primary_focal_length + 500 * u.mm,
                clear_radius=100 * u.mm,
            ),
            central_obscuration=components.CentralObscuration(
                piston=1404.270 * u.mm,
                obscured_radius=tuffet_radius,
                num_sides=num_sides,
            ),
            primary=components.Primary(
                radius=2 * primary_focal_length,
                num_sides=num_sides,
                clear_radius=primary_clear_radius,
                border_width=83.7 * u.mm - primary_clear_radius,
                substrate_thickness=30 * u.mm,
            ),
            field_stop=components.FieldStop(
                piston=primary_focal_length,
                clear_radius=1.82 * u.mm,
                mech_radius=2.81 * u.mm,
                num_sides=num_sides,
            ),
            grating=components.Grating(
                tangential_radius=grating_radius,
                sagittal_radius=grating_radius,
                groove_density=groove_density,
                piston=grating_piston,
                channel_radius=grating_channel_radius,
                channel_angle=channel_angle,
                inclination=-4.469567242792327 * u.deg,
                aper_half_angle=deg_per_channel / 2,
                aper_decenter_x=-grating_channel_radius,
                diffraction_order=1 << u.dimensionless_unscaled,
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
            filter=components.Filter(
                piston=filter_piston,
                channel_radius=95.9 * u.mm,
                channel_angle=channel_angle,
                inclination=-3.45 * u.deg,
                clear_radius=15.9 * u.mm,
            ),
            detector=components.Detector(
                piston=filter_piston - 200 * u.mm,
                channel_radius=108 * u.mm,
                channel_angle=channel_angle,
                inclination=-12.252 * u.deg,
                pix_half_width_x=pix_half_width,
                pix_half_width_y=pix_half_width,
                npix_x=2048,
                npix_y=1024,
            ),
        ),
        wavelengths=[629.7, 609.8, 584.3, ] * u.AA,
        field_limit=vector.from_components(field_limit, field_limit).to(u.arcmin),
        pupil_samples=pupil_samples,
        field_samples=field_samples,
    )


def final_from_poletto(
        pupil_samples: int = 10,
        field_samples: int = 10,
        use_toroidal_grating: bool = False,
        use_vls_grating: bool = False,
        use_one_wavelength_detector_tilt: bool = False,
) -> Optics:
    """
    Try to reproduce the final ESIS design using infrastructure developed from Thomas and Poletto (2004)
    :param pupil_samples: Number of rays across the pupil in each axis.
    :param field_samples: Number of rays across the field in each axis.
    :param use_toroidal_grating: Flag for choosing a toroidal or spherical grating.
    :param use_vls_grating: Flag for choosing uniform line spacing or variable line spacing.
    :param use_one_wavelength_detector_tilt: Flag for choosing whether to use one or two wavelengths to compute the
    detector tilt.
    :return: Instance of the ESIS optics model where the parameters have been calculated using Thomas and Poletto.
    """
    esis = final(pupil_samples=pupil_samples, field_samples=field_samples)

    obs_margin = esis.components.central_obscuration.obscured_radius - esis.components.grating.outer_clear_radius
    return esis.apply_poletto_layout(
        wavelength_1=esis.wavelengths[..., 0],
        wavelength_2=esis.wavelengths[..., ~0],
        magnification=4 * u.dimensionless_unscaled,
        obscuration_margin=obs_margin,
        image_margin=1 * u.mm,
        detector_is_opposite_grating=False,
        use_toroidal_grating=use_toroidal_grating,
        use_vls_grating=use_vls_grating,
    )

    # magnification = 4
    #
    # obscuration = esis.components.central_obscuration
    # primary = esis.components.primary
    # grating = esis.components.grating
    # detector = esis.components.detector
    #
    # new_grating = grating.from_gregorian_layout(
    #     magnification=magnification,
    #     primary_focal_length=primary.focal_length,
    #     primary_clear_radius=primary.clear_radius,
    #     detector_channel_radius=detector.channel_radius,
    #     detector_piston=detector.piston,
    #     obscuration_margin=obscuration.obscured_radius - grating.outer_clear_radius,
    # )
    #
    # grating.piston = new_grating.piston
    # grating.channel_radius = new_grating.channel_radius
    # grating.aper_decenter_x = new_grating.aper_decenter_x
    # grating.inner_clear_radius = new_grating.inner_clear_radius
    # grating.outer_clear_radius = new_grating.outer_clear_radius
    #
    # new_grating, new_detector = poletto.calc_grating_and_detector(
    #     wavelength_1=esis.wavelengths[..., 0],
    #     wavelength_2=esis.wavelengths[..., ~0],
    #     source_piston=primary.focal_length,
    #     magnification=magnification,
    #     grating_channel_radius=grating.channel_radius,
    #     grating_piston=grating.piston,
    #     detector_channel_radius=detector.channel_radius,
    #     use_toroidal_grating=use_toroidal_grating,
    #     use_vls_grating=use_vls_grating,
    #     use_one_wavelength_detector_tilt=use_one_wavelength_detector_tilt,
    # )
    #
    # grating.tangential_radius = new_grating.tangential_radius
    # grating.sagittal_radius = new_grating.sagittal_radius
    # grating.groove_density = new_grating.groove_density
    # grating.groove_density_coeff_linear = new_grating.groove_density_coeff_linear
    # grating.groove_density_coeff_quadratic = new_grating.groove_density_coeff_quadratic
    # grating.groove_density_coeff_cubic = new_grating.groove_density_coeff_cubic
    # grating.inclination = new_grating.inclination
    #
    # detector.piston = new_detector.piston
    # detector.inclination = new_detector.inclination
    #
    # esis.update()
    #
    # return esis
