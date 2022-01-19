import typing as typ
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.visualization
import pylatex
import kgpy.latex
import kgpy.format
import esis.optics
from .. import optics

__all__ = ['table_old']

table_old = r"""
\begin{table*}[!htb]
\caption{Imaging error budget and tolerance analysis results.  \MTF\ is given at \protect\SI{0.5}{cycles\per arcsecond}.}
\begin{tabular}{llrcc}
Element 	&	  		& Tolerance & $\sigma$ $[\mu m]$ & \MTF \\
\hline %-----------------------------------------------------------------------------
Primary M.	& Surface figure & (Ref. Table~\ref{table:surfaces}) & & 0.700 \\
			& Decenter	& 1 \si{\milli\meter}		& 1.700 	& 0.881 \\
            %& Defocus	& .08			& 3.9 		&  \\
Grating		& Surface figure & (Ref. Table~\ref{table:surfaces}) & & 0.600 \\
			& Radius	& 2.5 \si{\milli\meter}		& 1.410		& 0.916 \\
            & Decenter	& 1 \si{\milli\meter}		& 0.001		& 1.000 \\
            & Defocus	& 0.015 \si{\milli\meter}	& 0.801		& 0.972 \\
            & Clocking  & 13 \si{\milli\radian} 	& 1.300 	& 0.929 \\
CCD			& Decenter	& 1 \si{\milli\meter}		& 0.310		& 0.996 \\
			& Defocus 	& 0.229 \si{\milli\meter}	& 0.706 	& 0.978 \\            
\hline %-----------------------------------------------------------------------------
\multicolumn{2}{l}{Max RMS spot radius (modeled)} 	&				& 1.720			& 0.878 \\
\multicolumn{2}{l}{CCD charge diffusion (est.)} &				& 2.000			& 0.839 \\
Thermal drift	& 		&				& 0.192			& 0.998 \\
\SPARCSShort\ drift	& 		&				& 1.920			& 0.998 \\
Pointing jitter & 		&				& 3.430			& 0.597 \\
Diff. Limit 	& 		&				&				& 0.833 \\
\hline %-----------------------------------------------------------------------------
Total \MTF\	 	& 		&				&				& 0.109 \\
\end{tabular}
\label{table:tol}
\end{table*}"""

def table(doc: kgpy.latex.Document) -> pylatex.Table:
    requirements = esis.optics.design.requirements()
    optics_single = optics.as_designed_single_channel()
    opt = esis.optics.design.final(**optics.error_kwargs)

    wavelength = optics_single.bunch.wavelength
    index_o5 = np.nonzero(optics_single.bunch.ion == 'o_5')[0][0]
    wavelength_o5 = wavelength[index_o5]
    # opt = esis.optics.design.final(
    #     pupil_samples=101,
    #     # pupil_is_stratified_random=True,
    #     field_samples=11,
    #     all_channels=False,
    # )

    # psf_diffraction = opt.system.psf_diffraction

    # plt.figure()
    # plt.imshow(psf_diffraction.data[5,~0,..., 0].T, aspect='auto')
    #
    # print('position', psf_diffraction.grid.position)
    # print('position', psf_diffraction.grid.position.points.x.shape)
    # print('position', psf_diffraction.grid.position.points.y.shape)

    # opt.psf_diffraction
    opt.mtf_diffraction
    plt.show()

    # rays_grating = opt.system.raytrace[opt.system.surfaces_all.flat_local.index(opt.grating.surface)]
    # rays_grating.pupil_hist2d(bins=100)
    # plt.show()

    frequency_requirement = 1 / requirements.resolution_angular

    def calc_mtf(optics: esis.optics.Optics):
        rays = optics.rays_output
        mtf, frequency = rays.mtf(
            bins=200,
            frequency_min=frequency_requirement,
        )
        print('mtf', mtf.mean())
        print('frequency', frequency)
        mtf = np.take(a=mtf, indices=[index_o5], axis=rays.axis.wavelength)
        print('mtf', mtf.mean())
        mtf = np.mean(mtf.value, axis=rays.axis.field_xy, keepdims=True, where=mtf != 0) << mtf.unit
        print('mtf', mtf.mean())
        frequency = frequency.take(indices=[index_o5], axis=rays.axis.wavelength)
        frequency = np.mean(frequency, axis=rays.axis.field_xy, keepdims=True)

        plt.figure()
        plt.imshow(mtf.squeeze().T)

        with astropy.visualization.quantity_support():
            plt.figure()
            plt.plot(frequency.x.take(indices=0, axis=rays.axis.pupil_x).squeeze(),
                     mtf.take(indices=0, axis=rays.axis.pupil_x).squeeze())
            plt.plot(frequency.y.take(indices=0, axis=rays.axis.pupil_y).squeeze(),
                     mtf.take(indices=0, axis=rays.axis.pupil_y).squeeze())

        mtf = np.take(a=mtf, indices=[0], axis=rays.axis.pupil_x)
        print('mtf', mtf.mean())
        index_frequency_requirement = np.argmax(
            frequency.y.take(indices=[0], axis=rays.axis.pupil_x) >= frequency_requirement, axis=rays.axis.pupil_y)
        index_frequency_requirement = np.expand_dims(index_frequency_requirement, axis=rays.axis.pupil_y)
        print('index frequency requirement', index_frequency_requirement.shape)
        # mtf = np.take(a=mtf, indices=[index_frequency_requirement], axis=rays.axis.pupil_y)
        mtf = np.take_along_axis(mtf, indices=index_frequency_requirement, axis=rays.axis.pupil_y)
        print('mtf', mtf.mean())
        print('mtf.shape', mtf.shape)
        return mtf.squeeze()

    mtf_nominal = calc_mtf(opt)

    accumulator = dict(
        mtf=1 * u.dimensionless_unscaled,
    )

    def add_mtf(
            tabular: pylatex.Tabular,
            name_major: str = '',
            name_minor: str = '',
            value_str: str = '',
            mtf: u.Quantity = 0 * u.dimensionless_unscaled,
    ):
        accumulator['mtf'] *= mtf

        tabular.add_row([
            name_major,
            name_minor,
            value_str,
            f'',
            f'',
            f'{mtf.value:0.3f}',
        ])

    def add_optics(
            tabular: pylatex.Tabular,
            optics: typ.Union[esis.optics.Optics, typ.Tuple[esis.optics.Optics, esis.optics.Optics]],
            name_major: str = '',
            name_minor: str = '',
            value: typ.Optional[typ.Union[u.Quantity, typ.Tuple[u.Quantity, u.Quantity]]] = None,
            value_format_kwargs: typ.Optional[typ.Dict[str, typ.Any]] = None,
            remove_nominal_mtf: bool = True
    ):

        if value_format_kwargs is None:
            value_format_kwargs = dict(
                digits_after_decimal=3,
                scientific_notation=False,
            )

        if not isinstance(optics, esis.optics.Optics):
            optics_min, optics_max = optics

            mtf_min, mtf_max = calc_mtf(optics_min), calc_mtf(optics_max)

            if mtf_max < mtf_min:
                mtf = mtf_max
            else:
                mtf = mtf_min

            if value is not None:
                value_min, value_max = value
                if value_max == -value_min:
                    value_str = f'$\\pm${kgpy.format.quantity(value_max, **value_format_kwargs)}'
                else:
                    raise NotImplementedError
            else:
                value_str = ''

        else:
            mtf = calc_mtf(optics)
            if value is not None:
                value_str = f'{kgpy.format.quantity(value, **value_format_kwargs)}'
            else:
                value_str = ''

        if remove_nominal_mtf:
            mtf = mtf / mtf_nominal

        add_mtf(
            tabular=tabular,
            name_major=name_major,
            name_minor=name_minor,
            value_str=value_str,
            mtf=mtf,
        )

    opt = esis.optics.design.final(**optics.error_kwargs)
    # print(calc_mtf(opt_err))

    # rays_err = opt_err.rays_output
    # # rays_err.position.z = np.broadcast_to(rays_err.wavelength, rays_err.position.shape, subok=True).copy()
    # # rays_err.position = rays_err.distortion.model(inverse=True)(rays_err.position).to(u.arcsec)
    #
    # mtf, frequency = rays_err.mtf(
    #     bins=200,
    #     frequency_min=frequency_requirement,
    # )
    #
    # index_freq = np.argmax(frequency.y == frequency_requirement)
    # print('index_freq', index_freq)
    #
    # mtf[mtf == 0] = np.nan
    # mtf = np.nanmean(mtf, axis=(rays_err.axis.field_x, rays_err.axis.field_y))[..., 0, 0]
    # print('mtf', mtf[0, index_freq])
    #
    # print(mtf.shape)
    #
    # plt.figure()
    # plt.imshow(mtf.T)
    # # plt.show()
    #
    # with astropy.visualization.quantity_support():
    #     plt.figure()
    #     plt.plot(frequency.x, mtf[0])
    #     plt.plot(frequency.y, mtf[..., 0])
    #
    # plt.show()

    units_psf = u.pix
    plate_scale = optics_single.plate_scale
    focal_length_effective = optics_single.magnification.y * optics_single.primary.focal_length

    opt = esis.optics.design.final(**optics.error_kwargs)
    system_psf = np.nanmean(opt.rays_output.spot_size_rms[..., 0, :])

    frequency_mtf_arcsec = 0.5 * u.cycle / u.arcsec
    frequency_mtf = frequency_mtf_arcsec * plate_scale.y / u.cycle

    def to_mtf(psf_size: u.Quantity):
        psf_size = psf_size / np.sqrt(2)
        alpha = 1 / (2 * psf_size ** 2)
        return np.exp(-(np.pi * frequency_mtf) ** 2 / alpha)
        # return np.exp(-(2 * np.pi * frequency_mtf * psf_size) ** 2)

    def to_pix(value: u.Quantity):
        return value / (optics_single.detector.pixel_width / u.pix)

    def from_pix(value: u.Quantity):
        return value * (optics_single.detector.pixel_width / u.pix)

    primary_slope_error = optics_single.primary.slope_error.value
    primary_slope_error_psf = focal_length_effective * np.tan(2 * primary_slope_error)
    primary_slope_error_psf /= optics_single.detector.pixel_width / u.pix

    opt_primary_decenter_x_max = optics.error_primary_decenter_x_max()
    opt_primary_decenter_x_min = optics.error_primary_decenter_x_min()
    opt_primary_decenter_y_max = optics.error_primary_decenter_y_max()
    opt_primary_decenter_y_min = optics.error_primary_decenter_y_min()

    distance_grating_to_detector = (
                optics_single.detector.transform.translation_eff - optics_single.grating.transform.translation_eff).length
    grating_slope_error = optics_single.grating.slope_error.value
    grating_slope_error_psf = distance_grating_to_detector * np.tan(2 * grating_slope_error)
    grating_slope_error_psf /= optics_single.detector.pixel_width / u.pix

    opt_grating_translation_x_min = optics.error_grating_translation_x_min()
    opt_grating_translation_x_max = optics.error_grating_translation_x_max()
    opt_grating_translation_y_min = optics.error_grating_translation_y_min()
    opt_grating_translation_y_max = optics.error_grating_translation_y_max()
    opt_grating_translation_z_min = optics.error_grating_translation_z_min()
    opt_grating_translation_z_max = optics.error_grating_translation_z_max()
    opt_grating_roll_min = optics.error_grating_roll_min()
    opt_grating_roll_max = optics.error_grating_roll_max()
    opt_grating_radius_min = optics.error_grating_radius_min()
    opt_grating_radius_max = optics.error_grating_radius_max()
    opt_grating_ruling_density_min = optics.error_grating_ruling_density_min()
    opt_grating_ruling_density_max = optics.error_grating_ruling_density_max()
    opt_grating_ruling_spacing_linear_min = optics.error_grating_ruling_spacing_linear_min()
    opt_grating_ruling_spacing_linear_max = optics.error_grating_ruling_spacing_linear_max()
    opt_grating_ruling_spacing_quadratic_min = optics.error_grating_ruling_spacing_quadratic_min()
    opt_grating_ruling_spacing_quadratic_max = optics.error_grating_ruling_spacing_quadratic_max()

    opt_detector_translation_x_min = optics.error_detector_translation_x_min()
    opt_detector_translation_x_max = optics.error_detector_translation_x_max()
    opt_detector_translation_y_min = optics.error_detector_translation_y_min()
    opt_detector_translation_y_max = optics.error_detector_translation_y_max()
    opt_detector_translation_z_min = optics.error_detector_translation_z_min()
    opt_detector_translation_z_max = optics.error_detector_translation_z_max()

    rays = opt.system.rays_input.copy()
    rays.position = np.broadcast_to(rays.position, opt.rays_output.position.shape, subok=True).copy()
    rays.position[~opt.rays_output.mask] = np.nan
    rays_min = np.nanmin(rays.position, axis=(rays.axis.pupil_x, rays.axis.pupil_y))
    rays_max = np.nanmax(rays.position, axis=(rays.axis.pupil_x, rays.axis.pupil_y))
    rays_range = np.nanmean(rays_max - rays_min)
    detector_x = np.linspace(-1, 1, 100) / 2 * u.pix
    diffraction_intensity = np.sinc(rays_range.x / wavelength_o5 * u.rad * np.sin(detector_x * opt.plate_scale.x)) ** 2
    model = astropy.modeling.fitting.LevMarLSQFitter()(
        model=astropy.modeling.models.Gaussian1D(),
        x=detector_x,
        y=diffraction_intensity,
    )
    diffraction_limit = np.sqrt(2) * model.stddev.quantity

    accumulator = dict(
        psf_size_squared=0 * u.pix ** 2,
        mtf=1 * u.dimensionless_unscaled,
        mtf_actual=1 * u.dimensionless_unscaled,
    )

    def add_row_basic(
            tabular: pylatex.Tabular,
            optics: typ.Union[esis.optics.Optics, typ.Tuple[esis.optics.Optics, esis.optics.Optics]],
            name_major: str = '',
            name_minor: str = '',
            value_str: str = '',
            psf_size: u.Quantity = 0 * u.um,
            mtf_actual: u.Quantity = 1.0 * u.dimensionless_unscaled,
    ):

        mtf = to_mtf(psf_size)

        tabular.add_row([
            name_major,
            name_minor,
            value_str,
            f'{psf_size.to(u.pix).value:0.2f}',
            f'{(psf_size * optics.plate_scale.y).to(u.arcsec).value:0.2f}',
            f'{mtf.value:0.3f}',
            f'{mtf_actual.value:0.3f}',
        ])

        accumulator['psf_size_squared'] += np.square(psf_size)
        accumulator['mtf_actual'] *= mtf_actual
        accumulator['mtf'] *= mtf

    def add_row(
            tabular: pylatex.Tabular,
            optics: typ.Union[esis.optics.Optics, typ.Tuple[esis.optics.Optics, esis.optics.Optics]],
            name_major: str = '',
            name_minor: str = '',
            value: typ.Optional[typ.Union[u.Quantity, typ.Tuple[u.Quantity, u.Quantity]]] = None,
            digits_after_decimal: int = 3,
            scientific_notation: bool = False,
            remove_nominal_psf: bool = True,
    ):
        format_kwargs = dict(
            digits_after_decimal=digits_after_decimal,
            scientific_notation=scientific_notation,
        )

        if not isinstance(optics, esis.optics.Optics):
            optics_min, optics_max = optics

            psf_size_min = np.nanmean(optics_min.rays_output.spot_size_rms[..., 0, :])
            psf_size_max = np.nanmean(optics_max.rays_output.spot_size_rms[..., 0, :])

            if psf_size_max > psf_size_min:
                optics = optics_max
            else:
                optics = optics_min

            if value is not None:
                value_min, value_max = value
                if value_max == -value_min:
                    value_str = f'$\\pm${kgpy.format.quantity(value_max, **format_kwargs)}'
                else:
                    raise NotImplementedError
            else:
                value_str = ''

        else:
            if value is not None:
                value_str = f'{kgpy.format.quantity(value, **format_kwargs)}'
            else:
                value_str = ''

        psf_size = np.nanmean(optics.rays_output.spot_size_rms[..., 0, :])
        mtf_actual = calc_mtf(optics)
        print('mtf actual', mtf_actual)
        if remove_nominal_psf:
            psf_size = np.nan_to_num(np.sqrt(np.square(psf_size) - np.square(system_psf)))
            mtf_actual = mtf_actual / mtf_nominal

        add_row_basic(
            tabular=tabular,
            optics=optics,
            name_major=name_major,
            name_minor=name_minor,
            value_str=value_str,
            psf_size=psf_size,
            mtf_actual=mtf_actual,
        )

    def ptp_to_rms(value: u.Quantity) -> u.Quantity:
        return value / np.sqrt(8)

    result = pylatex.Table()
    result._star_latex_name = True
    with result.create(pylatex.Center()) as centering:
        with centering.create(pylatex.Tabular('ll|rrrrr')) as tabular:
            tabular.escape = False
            tabular.add_row([
                r'Element',
                r'',
                r'Tolerance',
                f'$\\sigma$ ({units_psf:latex_inline})',
                f'$\\sigma$ ({u.arcsec:latex_inline})',
                r'\MTF\ from $\sigma$',
                r'\MTF\ actual ',
            ])
            tabular.add_hline()
            add_row(
                tabular=tabular,
                optics=opt,
                name_major='System',
                name_minor='Aberration',
                remove_nominal_psf=False,
            )
            add_row_basic(
                tabular=tabular,
                optics=opt,
                name_minor='Diffraction',
                psf_size=diffraction_limit,
            )
            add_row_basic(
                tabular=tabular,
                optics=opt,
                name_minor='Thermal drift',
                psf_size=ptp_to_rms(opt.sparcs.pointing_drift / opt.plate_scale.x * opt.detector.exposure_length),
            )
            tabular.add_hline()
            add_row_basic(
                tabular=tabular,
                optics=opt,
                name_major='Primary',
                name_minor='RMS Slope error',
                value_str=f'{kgpy.format.quantity(primary_slope_error, digits_after_decimal=1)}',
                psf_size=primary_slope_error_psf,
                mtf_actual=opt.primary.mtf_degradation_factor,
            )
            add_row(
                tabular=tabular,
                optics=(
                    opt_primary_decenter_x_min,
                    opt_primary_decenter_x_max,
                ),
                name_minor='Translation $x$',
                value=(
                    -opt_primary_decenter_x_min.primary.translation_error.value.xy.length,
                    opt_primary_decenter_x_max.primary.translation_error.value.xy.length,
                ),
                digits_after_decimal=0,
            )
            add_row(
                tabular=tabular,
                optics=(
                    opt_primary_decenter_y_min,
                    opt_primary_decenter_y_max,
                ),
                name_minor='Translation $y$',
                value=(
                    -opt_primary_decenter_y_min.primary.translation_error.value.xy.length,
                    opt_primary_decenter_y_max.primary.translation_error.value.xy.length,
                ),
                digits_after_decimal=0,
            )
            tabular.add_hline()
            add_row_basic(
                tabular=tabular,
                optics=opt,
                name_major='Grating',
                name_minor='RMS Slope error',
                value_str=f'{kgpy.format.quantity(grating_slope_error, digits_after_decimal=1)}',
                psf_size=grating_slope_error_psf,
                mtf_actual=opt.grating.mtf_degradation_factor,
            )
            add_row(
                tabular=tabular,
                optics=(
                    opt_grating_translation_x_min,
                    opt_grating_translation_x_max,
                ),
                name_minor='Translation $x$',
                value=(
                    -opt_grating_translation_x_min.grating.translation_error.value.xy.length,
                    opt_grating_translation_x_max.grating.translation_error.value.xy.length,
                ),
                digits_after_decimal=0,
            )
            add_row(
                tabular=tabular,
                optics=(
                    opt_grating_translation_y_min,
                    opt_grating_translation_y_max,
                ),
                name_minor='Translation $y$',
                value=(
                    -opt_grating_translation_y_min.grating.translation_error.value.xy.length,
                    opt_grating_translation_y_max.grating.translation_error.value.xy.length,
                ),
                digits_after_decimal=0,
            )
            add_row(
                tabular=tabular,
                optics=(
                    opt_grating_translation_z_min,
                    opt_grating_translation_z_max,
                ),
                name_minor='Translation $z$',
                value=(
                    opt_grating_translation_z_min.grating.translation_error.z,
                    opt_grating_translation_z_max.grating.translation_error.z,
                ),
                digits_after_decimal=3,
            )
            add_row(
                tabular=tabular,
                optics=(
                    opt_grating_roll_min,
                    opt_grating_roll_max,
                ),
                name_minor='Roll',
                value=(
                    opt_grating_roll_min.grating.roll_error,
                    opt_grating_roll_max.grating.roll_error,
                ),
                digits_after_decimal=3,
            )
            add_row(
                tabular=tabular,
                optics=(
                    opt_grating_radius_min,
                    opt_grating_radius_max,
                ),
                name_minor='Radius',
                value=(
                    opt_grating_radius_min.grating.tangential_radius_error,
                    opt_grating_radius_max.grating.tangential_radius_error,
                ),
                digits_after_decimal=1,
            )
            add_row(
                tabular=tabular,
                optics=(
                    opt_grating_ruling_density_min,
                    opt_grating_ruling_density_max,
                ),
                name_minor='Ruling density',
                value=(
                    opt_grating_ruling_density_min.grating.ruling_density_error,
                    opt_grating_ruling_density_max.grating.ruling_density_error,
                ),
                digits_after_decimal=1,
            )
            add_row(
                tabular=tabular,
                optics=(
                    opt_grating_ruling_spacing_linear_min,
                    opt_grating_ruling_spacing_linear_max,
                ),
                name_minor='Linear coeff.',
                value=(
                    opt_grating_ruling_spacing_linear_min.grating.ruling_spacing_coeff_linear_error,
                    opt_grating_ruling_spacing_linear_max.grating.ruling_spacing_coeff_linear_error,
                ),
                digits_after_decimal=1,
                scientific_notation=True,
            )
            add_row(
                tabular=tabular,
                optics=(
                    opt_grating_ruling_spacing_quadratic_min,
                    opt_grating_ruling_spacing_quadratic_max,
                ),
                name_minor='Quadratic coeff.',
                value=(
                    opt_grating_ruling_spacing_quadratic_min.grating.ruling_spacing_coeff_quadratic_error,
                    opt_grating_ruling_spacing_quadratic_max.grating.ruling_spacing_coeff_quadratic_error,
                ),
                digits_after_decimal=1,
                scientific_notation=True,
            )
            tabular.add_hline()
            add_row(
                tabular=tabular,
                optics=(
                    opt_detector_translation_x_min,
                    opt_detector_translation_x_max,
                ),
                name_major='Detector',
                name_minor='Translation $x$',
                value=(
                    -opt_detector_translation_x_min.detector.translation_error.value.xy.length,
                    opt_detector_translation_x_max.detector.translation_error.value.xy.length,
                ),
                digits_after_decimal=0,
            )
            add_row(
                tabular=tabular,
                optics=(
                    opt_detector_translation_y_min,
                    opt_detector_translation_y_max,
                ),
                name_minor='Translation $y$',
                value=(
                    -opt_detector_translation_y_min.detector.translation_error.value.xy.length,
                    opt_detector_translation_y_max.detector.translation_error.value.xy.length,
                ),
                digits_after_decimal=0,
            )
            add_row(
                tabular=tabular,
                optics=(
                    opt_detector_translation_z_min,
                    opt_detector_translation_z_max,
                ),
                name_minor='Translation $z$',
                value=(
                    opt_detector_translation_z_min.detector.translation_error.z,
                    opt_detector_translation_z_max.detector.translation_error.z,
                ),
                digits_after_decimal=2,
            )
            add_row_basic(
                tabular=tabular,
                optics=opt,
                name_minor='Charge diffusion',
                psf_size=to_pix(opt.detector.charge_diffusion),
            )
            tabular.add_hline()
            add_row_basic(
                tabular=tabular,
                optics=opt,
                name_major=r'\SPARCSShort',
                name_minor='Pointing jitter',
                value_str=f'$\\pm${kgpy.format.quantity(opt.sparcs.pointing_jitter / 2, digits_after_decimal=2)}',
                psf_size=ptp_to_rms(opt.sparcs.pointing_jitter / opt.plate_scale.x),
            )
            add_row_basic(
                tabular=tabular,
                optics=opt,
                name_minor='Pointing drift',
                value_str=f'{kgpy.format.quantity(opt.sparcs.pointing_drift)}',
                psf_size=ptp_to_rms(opt.sparcs.pointing_drift / opt.plate_scale.x * opt.detector.exposure_length),
            )
            pointing = 10 * u.arcmin
            add_row_basic(
                tabular=tabular,
                optics=opt,
                name_minor='Roll jitter',
                value_str=f'$\\pm${kgpy.format.quantity(opt.sparcs.rlg_jitter / 2, digits_after_decimal=0)}',
                psf_size=ptp_to_rms(2 * np.sin(opt.sparcs.rlg_jitter / 2) * pointing / opt.plate_scale.x),
            )
            add_row_basic(
                tabular=tabular,
                optics=opt,
                name_minor='Roll drift',
                value_str=f'{kgpy.format.quantity(opt.sparcs.rlg_drift)}',
                psf_size=ptp_to_rms(2 * np.sin(
                    opt.sparcs.rlg_drift * opt.detector.exposure_length / 2) * pointing / opt.plate_scale.x),
            )
            tabular.add_hline()
            tabular.add_hline()
            psf_size_total = np.sqrt(accumulator['psf_size_squared'])
            doc.set_variable_quantity(
                name='spatialResolutionTotal',
                value=2 * psf_size_total * opt.plate_scale.x,
                digits_after_decimal=2,
            )
            add_row_basic(
                tabular=tabular,
                optics=opt,
                name_major='Total',
                psf_size=psf_size_total,
                mtf_actual=accumulator['mtf_actual'],
            )
    result.add_caption(pylatex.NoEscape(
        f"""
Imaging error budget and tolerance analysis results. 
\\MTF\\ is given at {kgpy.format.quantity(frequency_mtf_arcsec, digits_after_decimal=1)}."""
    ))
    result.append(kgpy.latex.Label('table:errorBudget'))
    return result
