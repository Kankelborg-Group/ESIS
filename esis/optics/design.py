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
import astropy.modeling
import kgpy.units
import kgpy.labeled
import kgpy.uncertainty
import kgpy.vectors
import kgpy.transforms
import kgpy.optics
import kgpy.nsroc
from . import Source, FrontAperture, CentralObscuration, Primary, FieldStop, Grating, Filter, Detector
from . import primary as module_primary
from . import optics as module_optics

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
        wavelength_samples: int = 3,
        velocity_samples: int = 1,
        pupil_samples: int = 11,
        pupil_is_stratified_random: bool = False,
        field_samples: int = 11,
        field_is_stratified_random: bool = False,
        all_channels: bool = True,
        use_uncertainty: bool = False,
) -> module_optics.Optics:
    """
    Final ESIS optical design prepared by Charles Kankelborg and Hans Courrier.
    :param pupil_samples: Number of rays per axis across the pupil.
    :param field_samples: Number of rays per axis across the field.
    :return: An instance of the as-designed ESIS optics model.
    """
    # axes = module_optics.OpticsAxes()

    num_sides = 8
    num_channels = 6

    channel_name = kgpy.labeled.Array(np.array([i for i in range(num_channels)]), axes=['channel'])
    if not all_channels:
        channel_name = channel_name[dict(channel=default_channel)]

    deg_per_channel = 360 * u.deg / num_sides
    channel_offset_angle = -deg_per_channel / 2
    channel_angle = kgpy.labeled.LinearSpace(
        start=0 * u.deg,
        stop=num_channels * deg_per_channel,
        num=num_channels,
        endpoint=False,
        axis='channel',
    )
    channel_angle = channel_angle + channel_offset_angle
    # channel_angle = channel_angle[::-1]
    if not all_channels:
        channel_angle = channel_angle[dict(channel=default_channel)]

    if not all_channels:
        roll = -channel_angle
    else:
        roll = 0 * u.deg

    dashstyle = (0, (1, 3))
    dashstyle_channels = kgpy.labeled.Array(
        array=np.array([dashstyle, 'solid', 'solid', 'solid', 'solid', dashstyle], dtype=object),
        axes=['channel'],
    )
    alpha_channels = kgpy.labeled.Array(np.array([0, 1, 1, 1, 1, 0]), axes=['channel'])

    primary = Primary()
    primary.radius = 2000 * u.mm
    primary.mtf_degradation_factor = 0.7 * u.dimensionless_unscaled
    primary.slope_error.value = 1.0 * u.urad
    primary.slope_error.length_integration = 4 * u.mm
    primary.slope_error.length_sample = 2 * u.mm
    primary.ripple.value = 2.5 * u.nm
    primary.ripple.periods_min = 0.06 * u.mm
    primary.ripple.periods_max = 6 * u.mm
    primary.microroughness.value = 1 * u.nm
    primary.microroughness.periods_min = 1.6 * u.um
    primary.microroughness.periods_max = 70 * u.um
    primary.num_sides = num_sides
    primary.clear_half_width = 77.9 * u.mm * np.cos(deg_per_channel / 2)
    primary.border_width = (83.7 * u.mm - primary.clear_radius) * np.cos(deg_per_channel / 2)
    primary.material = kgpy.optics.surfaces.materials.MeasuredMultilayerMirror()
    primary.material.thickness = 30 * u.mm
    primary.material.base.material = kgpy.labeled.Array(np.array(['Cr'], dtype=object), axes=['layer'])
    primary.material.base.thickness = kgpy.labeled.Array([5] * u.nm, axes=['layer'])
    primary.material.main.material = kgpy.labeled.Array(np.array(['SiC'], dtype=object), axes=['layer'])
    primary.material.main.thickness = kgpy.labeled.Array([25] * u.nm, axes=['layer'])
    primary.material.transmissivity_function = module_primary.efficiency.model.vs_wavelength()

    if use_uncertainty:
        primary.translation.x = kgpy.uncertainty.Uniform(primary.translation.x, width=1 * u.mm)
        primary.translation.y = kgpy.uncertainty.Uniform(primary.translation.y, width=1 * u.mm)

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
    grating.translation.z = -(primary.focal_length + 374.7 * u.mm)
    grating.translation_cylindrical.radius = 2.074999998438000e1 * u.mm
    grating.translation_cylindrical.azimuth = channel_angle.copy()
    grating.sagittal_radius = 597.830 * u.mm
    grating.tangential_radius = grating.sagittal_radius
    grating.mtf_degradation_factor = 0.6 * u.dimensionless_unscaled
    grating.slope_error.value = 3 * u.urad
    grating.slope_error.length_integration = 2 * u.mm
    grating.slope_error.length_sample = 1 * u.mm
    grating.ripple.value = 4 * u.nm
    grating.ripple.periods_min = 0.024 * u.mm
    grating.ripple.periods_max = 2.4 * u.mm
    grating.microroughness.value = 1 * u.nm
    grating.microroughness.periods_min = 0.02 * u.um
    grating.microroughness.periods_max = 2 * u.um
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
    grating.material.base.material = kgpy.labeled.Array(np.array(['Al'], dtype=object), axes=['layer'])
    grating.material.base.thickness = kgpy.labeled.Array([10] * u.nm, axes=['layer'])
    grating.material.main.material = kgpy.labeled.Array(np.array(['Mg', 'SiC', 'Al'], dtype=object), axes=['layer'])
    grating.material.main.thickness = kgpy.labeled.Array([30, 10, 1] * u.nm, axes=['layer'])
    grating.material.main.num_periods = 3
    grating.material.cap.material = kgpy.labeled.Array(np.array(['Mg', 'Al', 'SiC'], dtype=object), axes=['layer'])
    grating.material.cap.thickness = kgpy.labeled.Array([30, 4, 10] * u.nm, axes=['layer'])
    if all_channels:
        grating.plot_kwargs['linestyle'] = dashstyle_channels
        grating.plot_kwargs['alpha'] = alpha_channels

    if use_uncertainty:
        grating.translation.x = kgpy.uncertainty.Uniform(grating.translation.x, width=1 * u.mm)
        grating.translation.y = kgpy.uncertainty.Uniform(grating.translation.y, width=1 * u.mm)

        error_alignment_transfer_single_point = 2.5e-5 * u.m
        error_alignment_transfer_systematic = 5e-6 * u.m
        error_alignment_transfer = np.sqrt(error_alignment_transfer_single_point ** 2 / 3 + error_alignment_transfer_systematic ** 2).to(u.mm)
        grating.translation.z = kgpy.uncertainty.Uniform(grating.translation.z, width=error_alignment_transfer)

        roll_error = 1.3e-2 * u.rad
        grating.roll = kgpy.uncertainty.Uniform(grating.roll, width=roll_error.to(u.deg))

        radius_error = (0.4 * u.percent).to(u.dimensionless_unscaled)
        grating.tangential_radius = kgpy.uncertainty.Uniform(
            nominal=grating.tangential_radius,
            width=grating.tangential_radius * radius_error,
        )
        grating.sagittal_radius = kgpy.uncertainty.Uniform(
            nominal=grating.sagittal_radius,
            width=grating.sagittal_radius* radius_error,
        )

        grating.ruling_density = kgpy.uncertainty.Uniform(
            nominal=grating.ruling_density,
            width=1 / u.mm,
        )
        grating.ruling_spacing_coeff_linear = kgpy.uncertainty.Uniform(
            nominal=grating.ruling_spacing_coeff_linear,
            width=0.0512e-5 * u.um / u.mm,
        )
        grating.ruling_spacing_coeff_quadratic = kgpy.uncertainty.Uniform(
            nominal=grating.ruling_spacing_coeff_quadratic,
            width=0.08558e-7 * u.um / u.mm ** 2,
        )


    filter = Filter()
    filter.translation.z = grating.translation.z + 1.301661998854058 * u.m
    filter.translation_cylindrical.radius = 95.9 * u.mm
    filter.translation_cylindrical.azimuth = channel_angle.copy()
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
        filter.plot_kwargs['alpha'] = alpha_channels

    detector = Detector()
    detector.name = 'CCD230-42'
    detector.manufacturer = 'E2V'
    detector.translation.z = filter.translation.z + 200 * u.mm
    detector.translation_cylindrical.radius = 108 * u.mm
    detector.translation_cylindrical.azimuth = channel_angle.copy()
    detector.range_focus_adjustment = 13 * u.mm
    detector.inclination = -12.252 * u.deg
    detector.pixel_width = 15 * u.um
    detector.num_pixels = kgpy.vectors.Cartesian2D(2048, 1040)
    detector.npix_overscan = 2
    detector.npix_blank = 50
    detector.temperature = -55 * u.deg_C
    detector.gain = 1 * u.electron / u.adu
    detector.readout_noise = 4 * u.adu
    detector.position_ov = kgpy.vectors.Cartesian2D(x=7.2090754246099999 * u.mm, y=0 * u.mm)

    moses_pixel_width = 13.5 * u.um
    kernel = [
                 [.034, .077, .034],
                 [.076, .558, .076],
                 [.034, .077, .034],
             ] * u.dimensionless_unscaled
    fitter = astropy.modeling.fitting.LevMarLSQFitter()
    charge_diffusion_model = fitter(
        astropy.modeling.models.Gaussian2D(x_stddev=3 * u.um, y_stddev=3 * u.um),
        *np.broadcast_arrays(
            np.linspace(-1, 1, 3)[:, np.newaxis] * moses_pixel_width,
            np.linspace(-1, 1, 3)[np.newaxis, :] * moses_pixel_width,
            kernel,
            subok=True,
        )
    )
    detector.charge_diffusion = kgpy.vectors.Cartesian2D(
        x=charge_diffusion_model.x_stddev.quantity,
        y=charge_diffusion_model.y_stddev.quantity,
    ).length

    detector.time_frame_transfer = 49.598 * u.ms
    detector.time_readout = 1.1 * u.s
    detector.exposure_length = 10 * u.s
    detector.exposure_length_min = 2 * u.s
    detector.exposure_length_max = 600 * u.s
    detector.exposure_length_increment = 100 * u.ms
    detector.bits_analog_to_digital = 16
    detector.error_synchronization = 1 * u.ms
    if all_channels:
        detector.index_trigger = 1
        detector.plot_kwargs['linestyle'] = dashstyle_channels
        detector.plot_kwargs['alpha'] = alpha_channels

    if use_uncertainty:
        detector.translation.x = kgpy.uncertainty.Uniform(detector.translation.x, width=1 * u.mm)
        detector.translation.y = kgpy.uncertainty.Uniform(detector.translation.y, width=1 * u.mm)

        uncertainty_focus = 5e-4 * u.m
        shim_length = 5e-5 * u.m
        edge_defocus = 1e-4 * u.m
        detector_defocus = np.sqrt(np.square(uncertainty_focus) + np.square(shim_length) + np.square(edge_defocus))
        detector_defocus = detector_defocus.to(u.mm)
        detector.translation.z = kgpy.uncertainty.Uniform(detector.translation.z, width=detector_defocus)

    # field_limit = (0.09561 * u.deg).to(u.arcmin)
    # if all_channels:
    #     field_limit = np.arctan2(field_stop.clear_radius, primary.focal_length).to(u.arcmin)
    # else:
    #     field_limit = np.arctan2(field_stop.clear_width / 2, primary.focal_length).to(u.arcmin)
    source = Source()
    source.piston = front_aperture.piston + 100 * u.mm
    # source.half_width_x = field_limit
    # source.half_width_y = field_limit

    # if pupil_is_stratified_random:
    #     type_grid_pupil = kgpy.grid.StratifiedRandomGrid2D
    # else:
    #     type_grid_pupil = kgpy.grid.RegularGrid2D
    #
    # if field_is_stratified_random:
    #     type_grid_field = kgpy.grid.StratifiedRandomGrid2D
    # else:
    #     type_grid_field = kgpy.grid.RegularGrid2D

    if isinstance(pupil_samples, int):
        pupil_samples = kgpy.vectors.Cartesian2D(x=pupil_samples, y=pupil_samples)

    if isinstance(field_samples, int):
        field_samples = kgpy.vectors.Cartesian2D(x=field_samples, y=field_samples)

    object_grid_normalized = kgpy.optics.vectors.ObjectVector(
        wavelength_base=kgpy.labeled.LinearSpace(0, 1, num=wavelength_samples, axis='spectral_line'),
        wavelength_offset=kgpy.labeled.LinearSpace(-1 * u.AA, 1 * u.AA, num=velocity_samples, axis='velocity'),
        field_x=kgpy.labeled.LinearSpace(-1, 1, num=field_samples.x, axis='field_x'),
        field_y=kgpy.labeled.LinearSpace(-1, 1, num=field_samples.y, axis='field_y'),
        pupil_x=kgpy.labeled.LinearSpace(-1, 1, num=pupil_samples.x, axis='pupil_x'),
        pupil_y=kgpy.labeled.LinearSpace(-1, 1, num=pupil_samples.y, axis='pupil_y'),

        # velocity_los=kgpy.labeled.LinearSpace(-100 * u.km / u.s, 100 * u.km / u.s, num=3, axis='velocity_los')
    )

    return module_optics.Optics(
        name='ESIS',
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
        object_grid_normalized=object_grid_normalized,
        sparcs=kgpy.nsroc.sparcs.specification(),
        transform_pointing=kgpy.transforms.TransformList([
            kgpy.transforms.RotationZ(roll),
        ]),
        skin_diameter=22 * u.imperial.inch,
    )

channels_active_min = 1
channels_active_max = 4
slice_channels_active = dict(channel=slice(channels_active_min, channels_active_max + 1))
default_channel_active = default_channel - channels_active_min


def final_active(
        wavelength_samples: int = 3,
        velocity_samples: int = 1,
        pupil_samples: int = 11,
        pupil_is_stratified_random: bool = False,
        field_samples: int = 11,
        field_is_stratified_random: bool = False,
) -> module_optics.Optics:
    opt = final(
        wavelength_samples=wavelength_samples,
        velocity_samples=velocity_samples,
        pupil_samples=pupil_samples,
        pupil_is_stratified_random=pupil_is_stratified_random,
        field_samples=field_samples,
        field_is_stratified_random=field_is_stratified_random,
    )

    opt.channel_name = opt.channel_name[slice_channels_active]

    opt.grating.translation_cylindrical.azimuth = opt.grating.translation_cylindrical.azimuth[slice_channels_active]
    opt.grating.plot_kwargs['linestyle'] = None
    opt.grating.plot_kwargs['alpha'] = 1

    opt.filter.translation_cylindrical.azimuth = opt.filter.translation_cylindrical.azimuth[slice_channels_active]
    opt.filter.plot_kwargs['linestyle'] = None
    opt.filter.plot_kwargs['alpha'] = 1

    opt.detector.translation_cylindrical.azimuth = opt.detector.translation_cylindrical.azimuth[slice_channels_active]
    opt.detector.plot_kwargs['linestyle'] = None
    opt.detector.plot_kwargs['alpha'] = 1
    opt.detector.index_trigger = opt.detector.index_trigger - slice_channels_active['channel'].start

    return opt


def final_from_poletto(
        pupil_samples: int = 10,
        field_samples: int = 10,
        use_toroidal_grating: bool = False,
        use_vls_grating: bool = False,
        use_one_wavelength_detector_tilt: bool = False,
) -> module_optics.Optics:
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
