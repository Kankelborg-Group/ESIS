r"""

.. jupyter-execute::

    import matplotlib.pyplot as plt
    import matplotlib.colors
    import IPython.display
    from kgpy import vector, optics
    import esis

    IPython.display.set_matplotlib_formats('svg')

Layout
------

ESIS is an array of slitless spectrographs, each with a different dispersion direction, but all fed from the same
primary mirror.

Each slitless spectrograph is an off-axis gregorian telescope.

The layout of a single slitless spectrograph (known as a channel) is shown in the figure below.
In this diagram, rays from the Sun enter on the left-hand side, reflect off the parabolic primary mirror on the
right-hand side and are focused onto the field stop in the center of the diagram.
After the field stop, the rays reflect off the grating on the left, and are refocused onto the detector on the
bottom-right and dispersed according to their wavelength.


.. jupyter-execute::

    _, ax_top = plt.subplots(figsize=(9.5, 4), constrained_layout=True)
    opt_top = esis.optics.design.final(field_samples=1, pupil_samples=3, all_channels=False)
    opt_top.system.plot(
        ax=ax_top,
        components=(vector.iz, vector.ix),
        plot_vignetted=True,
    )
    _ = ax_top.set_title('Top View, Channel 0 only')


.. jupyter-execute::

    _, ax_side = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax_side.invert_xaxis()
    esis.optics.design.final().system.plot(
        ax=ax_side,
        plot_rays=False,
    )
    _ = ax_side.set_title('Front View')
    ax_side.set_aspect('equal')


Ideal Point-spread Function
---------------------------

.. jupyter-execute::

    rays_psf = esis.optics.design.final(
        pupil_samples=101,
        field_samples=5,
        all_channels=False
    ).rays_output
    bins = rays_psf.input_pupil_x.shape[~0] // 2

    fig_630 = rays_psf.plot_pupil_hist2d_vs_field(wavlen_index=~0, norm=matplotlib.colors.PowerNorm(1/2), bins=bins, )
    fig_630.set_figheight(4)
    fig_630.set_figwidth(9.5)

    fig_584 = rays_psf.plot_pupil_hist2d_vs_field(wavlen_index=0, norm=matplotlib.colors.PowerNorm(1/2), bins=bins, )
    fig_584.set_figheight(4)
    fig_584.set_figwidth(9.5)



Vignetting
----------

.. jupyter-execute::

    rays = esis.optics.design.final(
        pupil_samples=21,
        field_samples=21,
        all_channels=False
    ).rays_output

    vignetting_linear = rays.vignetting(polynomial_degree=1)
    vignetting_linear.model().dataframe.to_html()

Distortion
----------

.. math::

   (a + b)^2  &=  (a + b)(a + b) \\
              &=  a^2 + 2ab + b^2

"""
import dataclasses

import numpy as np
from astropy import units as u
import kgpy.units
from kgpy import Name, vector
from . import Source, FrontAperture, CentralObscuration, Primary, FieldStop, Grating, Filter, Detector, Optics

__all__ = [
    'Requirements',
    'requirements',
    'default_channel',
    'final',
    'final_active',
    'final_from_poletto',
]


@dataclasses.dataclass
class Requirements:
    resolution_spatial: u.Quantity
    resolution_spectral: u.Quantity
    fov: u.Quantity
    snr: u.Quantity
    cadence: u.Quantity
    length_observation: u.Quantity

    @property
    def resolution_angular(self) -> u.Quantity:
        return np.arctan2(self.resolution_spatial, 1 * u.AU).to(u.arcsec)


def requirements() -> Requirements:
    return Requirements(
        resolution_spatial=1.5 * u.Mm,
        resolution_spectral=18 * u.km / u.s,
        fov=10 * u.arcmin,
        snr=17.3 * u.dimensionless_unscaled,
        cadence=15 * u.s,
        length_observation=150 * u.s,
    )


default_channel = 1


def final(
        pupil_samples: int = 10,
        pupil_is_stratified_random: bool = False,
        field_samples: int = 10,
        field_is_stratified_random: bool = False,
        all_channels: bool = True,
) -> Optics:
    """
    Final ESIS optical design prepared by Charles Kankelborg and Hans Courrier.
    :param pupil_samples: Number of rays per axis across the pupil.
    :param field_samples: Number of rays per axis across the field.
    :return: An instance of the as-designed ESIS optics model.
    """
    num_sides = 8
    num_channels = 6

    channel_name = np.array([i for i in range(num_channels)])
    if not all_channels:
        channel_name = channel_name[default_channel]

    deg_per_channel = 360 * u.deg / num_sides
    channel_offset_angle = -deg_per_channel / 2
    channel_angle = np.linspace(0 * u.deg, num_channels * deg_per_channel, num_channels, endpoint=False)
    channel_angle += channel_offset_angle
    # channel_angle = channel_angle[::-1]
    if not all_channels:
        channel_angle = channel_angle[default_channel]

    if not all_channels:
        roll = -channel_angle
    else:
        roll = 0 * u.deg

    dashstyle = (0, (1, 1))
    dashstyle_channels = np.array([dashstyle, None, None, None, None, dashstyle], dtype=object)

    primary = Primary()
    primary.radius = 2000 * u.mm
    primary.num_sides = num_sides
    primary.clear_half_width = 77.9 * u.mm * np.cos(deg_per_channel / 2)
    primary.border_width = (83.7 * u.mm - primary.clear_radius) * np.cos(deg_per_channel / 2)
    primary.material.thickness = 30 * u.mm
    primary.material.base.material = np.array(['Cr'])
    primary.material.base.thickness = [5] * u.nm
    primary.material.main.material = np.array(['SiC'])
    primary.material.main.thickness = [25] * u.nm

    front_aperture = FrontAperture()
    front_aperture.piston = primary.focal_length + 500 * u.mm
    front_aperture.clear_radius = 100 * u.mm

    tuffet_x1, tuffet_y1 = 2.54 * u.mm, 37.1707 * u.mm
    tuffet_x2, tuffet_y2 = 24.4876 * u.mm, 28.0797 * u.mm
    tuffet_dx, tuffet_dy = tuffet_x2 - tuffet_x1, tuffet_y2 - tuffet_y1
    tuffet_slope = tuffet_dy / tuffet_dx
    tuffet_radius = tuffet_y1 - tuffet_slope * tuffet_x1
    central_obscuration = CentralObscuration()
    central_obscuration.piston = 1404.270 * u.mm
    central_obscuration.obscured_half_width = tuffet_radius * np.cos(deg_per_channel / 2)

    field_stop = FieldStop()
    field_stop.piston = primary.focal_length.copy()
    field_stop.clear_radius = 1.82 * u.mm
    field_stop.mech_radius = 2.81 * u.mm
    field_stop.num_sides = num_sides

    grating = Grating()
    grating.piston = primary.focal_length + 374.7 * u.mm
    grating.cylindrical_radius = 2.074999998438000e1 * u.mm
    grating.cylindrical_azimuth = channel_angle.copy()
    grating.sagittal_radius = 597.830 * u.mm
    grating.tangential_radius = grating.sagittal_radius
    grating.nominal_input_angle = 1.301 * u.deg
    grating.nominal_output_angle = 8.057 * u.deg
    grating.ruling_density = (2.586608603456000 / u.um).to(1 / u.mm)
    grating.inclination = -4.469567242792327 * u.deg
    grating.aper_wedge_angle = deg_per_channel
    grating.diffraction_order = 1 * u.dimensionless_unscaled
    grating.ruling_spacing_coeff_linear = -3.3849e-5 * (u.um / u.mm)
    grating.ruling_spacing_coeff_quadratic = -1.3625e-7 * (u.um / u.mm ** 2)
    # d0 = 1 / grating.ruling_density
    # d_c1 = -3.3849e-5 * (u.um / u.mm)
    # d_c2 = -1.3625e-7 * (u.um / u.mm ** 2)
    # grating.ruling_density_coeff_linear = -d_c1 / np.square(d0)
    # grating.ruling_density_coeff_quadratic = (np.square(d_c1) - d0 * d_c2) / np.power(d0, 3)
    grating.border_width = 2 * u.mm
    grating.inner_border_width = 4.86 * u.mm
    grating.inner_half_width = 13.02 * u.mm - grating.inner_border_width
    grating.outer_half_width = 10.49 * u.mm - grating.border_width
    grating.dynamic_clearance = 1.25 * u.mm
    grating.material.thickness = -10 * u.mm
    grating.material.base.material = np.array(['Al'])
    grating.material.base.thickness = [10] * u.nm
    grating.material.main.material = np.array(['Mg', 'SiC', 'Al'])
    grating.material.main.thickness = [30, 10, 1] * u.nm
    grating.material.main.num_periods = 3
    grating.material.cap.material = np.array(['Mg', 'Al', 'SiC'])
    grating.material.cap.thickness = [30, 4, 10] * u.nm
    if all_channels:
        grating.plot_kwargs['linestyle'] = dashstyle_channels

    filter = Filter()
    filter.piston = grating.piston - 1.301661998854058 * u.m
    filter.cylindrical_radius = 95.9 * u.mm
    filter.cylindrical_azimuth = channel_angle.copy()
    filter.inclination = -3.45 * u.deg
    filter.clocking = 45 * u.deg
    filter.clear_radius = 15 * u.mm
    filter.thickness = 100 * u.nm
    filter.thickness_oxide = 4 * u.nm
    filter.mesh_ratio = 82 * u.percent
    filter.mesh_pitch = 70 * kgpy.units.line / u.imperial.inch
    filter.mesh_material = 'Ni'
    if all_channels:
        filter.plot_kwargs['linestyle'] = dashstyle_channels

    detector = Detector()
    detector.name = Name('CCD230-42')
    detector.manufacturer = 'E2V'
    detector.piston = filter.piston - 200 * u.mm
    detector.cylindrical_radius = 108 * u.mm
    detector.cylindrical_azimuth = channel_angle.copy()
    detector.inclination = -12.252 * u.deg
    detector.pixel_width = 15 * u.um
    detector.num_pixels = (2048, 1040)
    detector.npix_overscan = 2
    detector.npix_blank = 50
    detector.gain = 1 * u.electron / u.adu
    detector.readout_noise = 4 * u.adu
    detector.time_frame_transfer = 60 * u.ms
    detector.time_readout = 1.1 * u.s
    detector.exposure_length_min = 1.2 * u.s
    detector.bits_analog_to_digital = 16
    if all_channels:
        detector.plot_kwargs['linestyle'] = dashstyle_channels

    field_limit = (0.09561 * u.deg).to(u.arcmin)
    if all_channels:
        field_limit = np.arctan2(field_stop.clear_radius, primary.focal_length).to(u.arcmin)
    else:
        field_limit = np.arctan2(field_stop.clear_width / 2, primary.focal_length).to(u.arcmin)
    source = Source()
    source.piston = front_aperture.piston + 100 * u.mm
    source.half_width_x = field_limit
    source.half_width_y = field_limit

    return Optics(
        name=Name('ESIS'),
        channel_name=channel_name,
        source=source,
        front_aperture=front_aperture,
        central_obscuration=central_obscuration,
        primary=primary,
        field_stop=field_stop,
        grating=grating,
        filter=filter,
        detector=detector,
        num_emission_lines=10,
        pupil_samples=pupil_samples,
        pupil_is_stratified_random=pupil_is_stratified_random,
        field_samples=field_samples,
        field_is_stratified_random=field_is_stratified_random,
        roll=roll,
    )

channels_active_min = 1
channels_active_max = 4
slice_channels_active = slice(channels_active_min, channels_active_max + 1)
default_channel_active = default_channel - channels_active_min


def final_active(
        pupil_samples: int = 10,
        pupil_is_stratified_random: bool = False,
        field_samples: int = 10,
        field_is_stratified_random: bool = False,
) -> Optics:
    opt = final(
        pupil_samples=pupil_samples,
        pupil_is_stratified_random=pupil_is_stratified_random,
        field_samples=field_samples,
        field_is_stratified_random=field_is_stratified_random,
    )

    opt.channel_name = opt.channel_name[slice_channels_active]

    opt.grating.cylindrical_azimuth = opt.grating.cylindrical_azimuth[slice_channels_active]
    opt.grating.plot_kwargs['linestyle'] = None

    opt.filter.cylindrical_azimuth = opt.filter.cylindrical_azimuth[slice_channels_active]
    opt.filter.plot_kwargs['linestyle'] = None

    opt.detector.cylindrical_azimuth = opt.detector.cylindrical_azimuth[slice_channels_active]
    opt.detector.plot_kwargs['linestyle'] = None

    return opt


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

    obscuration = esis.central_obscuration
    grating = esis.grating
    obs_thickness = obscuration.piston - grating.piston
    obs_margin = obscuration.obscured_half_width - (grating.cylindrical_radius + grating.outer_half_width)
    primary_clear_radius = esis.primary.surface.aperture.min_radius
    detector = esis.detector
    detector_radius = detector.cylindrical_radius - detector.surface.aperture.half_width_x
    detector.dynamic_clearance = detector_radius - primary_clear_radius
    return esis.apply_poletto_layout(
        wavelength_1=esis.wavelength[..., 0],
        wavelength_2=esis.wavelength[..., ~0],
        magnification=4 * u.dimensionless_unscaled,
        obscuration_margin=obs_margin,
        obscuration_thickness=obs_thickness,
        image_margin=67 * detector.pixel_width,
        detector_is_opposite_grating=False,
        use_toroidal_grating=use_toroidal_grating,
        use_vls_grating=use_vls_grating,
    )
