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

    primary = components.Primary(
        radius=2000 * u.mm,
        num_sides=num_sides,
        clear_radius=77.9 * u.mm,
        substrate_thickness=30 * u.mm,
    )
    primary.border_width = 83.7 * u.mm - primary.clear_radius

    front_aperture = components.FrontAperture(
        piston=primary.focal_length + 500 * u.mm,
        clear_radius=100 * u.mm,
    )

    tuffet_x1, tuffet_y1 = 2.54 * u.mm, 37.1707 * u.mm
    tuffet_x2, tuffet_y2 = 24.4876 * u.mm, 28.0797 * u.mm
    tuffet_dx, tuffet_dy = tuffet_x2 - tuffet_x1, tuffet_y2 - tuffet_y1
    tuffet_slope = tuffet_dy / tuffet_dx
    tuffet_radius = tuffet_y1 - tuffet_slope * tuffet_x1
    central_obscuration = components.CentralObscuration(
        piston=1404.270 * u.mm,
        obscured_radius=tuffet_radius,
        num_sides=num_sides,
    )

    field_stop = components.FieldStop(
        piston=primary.focal_length.copy(),
        clear_radius=1.82 * u.mm,
        mech_radius=2.81 * u.mm,
        num_sides=num_sides,
    )

    grating = components.Grating(
        piston=primary.focal_length + 374.7 * u.mm,
        cylindrical_radius=2.074999998438000e1 * u.mm,
        cylindrical_azimuth=channel_angle.copy(),
        sagittal_radius=597.830 * u.mm,
        nominal_input_angle=1.301 * u.deg,
        nominal_output_angle=8.057 * u.deg,
        groove_density=2.586608603456000 / u.um,
        inclination=-4.469567242792327 * u.deg,
        aper_half_angle=deg_per_channel / 2,
        diffraction_order=1 << u.dimensionless_unscaled,
        inner_border_width=4.86 * u.mm,
        outer_border_width=2 * u.mm,
        side_border_width=2 * u.mm,
        dynamic_clearance=1.25 * u.mm,
        substrate_thickness=10 * u.mm,
    )
    grating.tangential_radius = grating.sagittal_radius
    grating.aper_decenter_x = -grating.cylindrical_radius
    d0 = 1 / grating.ruling_density
    d_c1 = -3.3849e-5 * (u.um / u.mm)
    d_c2 = -1.3625e-7 * (u.um / u.mm ** 2)
    grating.ruling_density_coeff_linear = -d_c1 / np.square(d0)
    grating.ruling_density_coeff_quadratic = (np.square(d_c1) - d0 * d_c2) / np.power(d0, 3)
    grating.inner_clear_radius = grating.cylindrical_radius - (13.02 * u.mm - grating.inner_border_width)
    grating.outer_clear_radius = grating.cylindrical_radius + (10.49 * u.mm - grating.outer_border_width)

    filter = components.Filter(
        piston=grating.piston - 1.301661998854058 * u.m,
        cylindrical_radius=95.9 * u.mm,
        cylindrical_azimuth=channel_angle.copy(),
        inclination=-3.45 * u.deg,
        clear_radius=15.9 * u.mm,
    )

    detector = components.Detector(
        piston=filter.piston - 200 * u.mm,
        cylindrical_radius=108 * u.mm,
        cylindrical_azimuth=channel_angle.copy(),
        inclination=-12.252 * u.deg,
        pixel_width=15 * u.um,
        num_pixels=(2048, 1024),
    )

    field_limit = 0.09561 * u.deg
    return Optics(
        name=Name('ESIS'),
        components=components.Components(
            front_aperture=front_aperture,
            central_obscuration=central_obscuration,
            primary=primary,
            field_stop=field_stop,
            grating=grating,
            filter=filter,
            detector=detector,
        ),
        wavelengths=[629.7, 609.8, 584.3, ] * u.AA,
        field_half_width=vector.from_components(field_limit, field_limit).to(u.arcmin),
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

    obs_thickness = esis.components.central_obscuration.piston - esis.components.grating.piston
    obs_margin = esis.components.central_obscuration.obscured_radius - esis.components.grating.outer_clear_radius
    primary_clear_radius = esis.components.primary.surface.aperture.min_radius
    detector_radius = esis.components.detector.channel_radius - esis.components.detector.surface.aperture.half_width_x
    esis.components.detector.dynamic_clearance = detector_radius - primary_clear_radius
    return esis.apply_poletto_layout(
        wavelength_1=esis.wavelengths[..., 0],
        wavelength_2=esis.wavelengths[..., ~0],
        magnification=4 * u.dimensionless_unscaled,
        obscuration_margin=obs_margin,
        obscuration_thickness=obs_thickness,
        image_margin=67 * 2 * esis.components.detector.pix_half_width_x,
        detector_is_opposite_grating=False,
        use_toroidal_grating=use_toroidal_grating,
        use_vls_grating=use_vls_grating,
    )
