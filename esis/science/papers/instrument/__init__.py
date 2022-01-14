import typing as typ
import pathlib
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.modeling
import astropy.visualization
import numpy as np
import pylatex
import kgpy.format
import kgpy.latex
import kgpy.units
import kgpy.chianti
import kgpy.optics
import esis.optics
import esis.science.papers.instrument.figures as figures
from . import optics
from . import preamble
from . import variables
from . import authors
from . import tables
from . import sections

path_base = pathlib.Path(__file__).parent
path_pdf = path_base / 'instrument'
path_figures = path_base / 'figures'


def document() -> kgpy.latex.Document:

    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 9
    plt.rcParams['lines.linewidth'] = 1

    doc = kgpy.latex.Document(
        default_filepath=str(path_pdf),
        documentclass='aastex631',
        document_options=[
            'twocolumn',
            # 'linenumbers',
        ]
    )

    doc.packages.append(pylatex.Package('paralist'))
    doc.packages.append(pylatex.Package('amsmath'))
    doc.packages.append(pylatex.Package('acronym'))
    doc.packages.append(pylatex.Package('savesym'))

    doc.preamble += preamble.body()

    variables.append_to_document(doc)

    doc.append(kgpy.latex.Title('The EUV Snapshot Imaging Spectrograph'))

    doc += authors.author_list()

    requirements = esis.optics.design.requirements()
    optics_single = optics.as_designed_single_channel()
    optics_all = esis.flight.optics.as_measured()
    wavelength = optics_single.bunch.wavelength
    index_o5 = np.nonzero(optics_single.bunch.ion == 'o_5')[0][0]
    wavelength_o5 = wavelength[index_o5]
    index_mg10_2 = np.nonzero(optics_single.bunch.ion == 'mg_10')[0][1]
    wavelength_mg10_2 = wavelength[index_mg10_2]

    doc.append(sections.abstract.section())

    doc.append(sections.introduction.section())

    doc.append(sections.esis_concept.section())

    doc.append(sections.science_objectives.section())

    with doc.create(pylatex.Section(pylatex.NoEscape('The \ESIS\ Instrument'))):
        doc.append(pylatex.NoEscape(
            r"""\ESIS\ is a multi-projection slitless spectrograph that obtains line intensities, Doppler shifts, and 
widths in a single snapshot over a 2D \FOV.
Starting from the notional instrument described in Sec.~\ref{sec:TheESISConcept}, \ESIS\ has been designed to ensure all 
of the science requirements set forth in Table~\ref{table:scireq} are met.
The final design parameters are summarized in Table~\ref{table:prescription}.

A schematic diagram of a single \ESIS\ channel is presented in Fig.~\ref{fig:schematic}a, while the mechanical features 
of the primary mirror and gratings are detailed in Figs.~\ref{fig:schematic}b and \ref{fig:schematic}c, respectively."""
        ))

        doc.append(tables.prescription.table())

        doc.append(sections.esis_instrument.optics.subsection())

        with doc.create(pylatex.Subsection('Optimization and Tolerancing')):
            doc.append(pylatex.NoEscape(
                r"""The science resolution requirement of \angularResolutionRequirement (Table~\ref{table:scireq}) was 
flowed down to specifications for the \ESIS\ optics.
To ensure that \ESIS\ meets this requirement, an imaging error budget was developed to track parameters that 
significantly influence instrument resolution.
The budget is roughly divided into two categories;
the first includes `variable' parameters that can be directly controlled (\eg, the figure and finish of the optics, 
grating radius and ruling, placement of the elements in the system, and the accuracy to which the instrument is 
focused).
The second category consists of `fixed' contributions (\eg, \CCD\ charge diffusion, pointing stability, and diffraction 
from the entrance aperture).
In this sub-section we describe the optimization of the first category to balance the contributions of the second. 

Figure and surface roughness specifications for the primary mirror and gratings were developed first by a rule of thumb 
and then validated through a Fourier optics based model \roy{Fourier-optics-based model} and Monte Carlo simulations.
Surface figure errors were randomly generated, using a power law distribution in frequency.
The model explored a range of power spectral distributions for the surface figure errors, with power laws ranging from 
0.1 to 4.0.
For each randomly generated array of optical figure errors, the amplitude was adjusted to yield a target \MTF\ 
degradation factor, as compared to the diffraction limited \roy{diffraction-limited} \MTF.
For the primary mirror, the figure of merit was a \MTF\ degradation of 0.7 \roy{\primaryMtfDegradationFactor} at \angularResolutionRequirement\ resolution.
Though the grating is smaller and closer to the focal plane, it was allocated somewhat more significant \MTF\ 
degradation of 0.6 \roy{\gratingMtfDegradationFactor} based on manufacturing capabilities.
The derived requirements are described in table~\ref{table:error}.
Note that this modeling exercise was undertaken before the baffle designs were finalized.
The estimated diffraction \MTF\ and aberrations were therefore modeled for a rough estimate of the \ESIS\ single sector 
aperture."""
            ))

            doc.append(tables.surface_error.table_old())

            doc.append(tables.surface_error.table())

            doc.append(pylatex.NoEscape(
                r"""The initial grating radius of curvature, $R_g$, and ruling pattern of the \ESIS\ gratings were 
derived from the analytical equations developed by \citet{Poletto04} for stigmatic spectrometers.
A second order polynomial describes the ruling pattern,
\begin{equation} \label{Eq-d}
    d = d_0 + d_1 r + d_2 r^2 \, ,
\end{equation}
where $r$ runs radially outward from the optical axis with its origin at the center of the grating \roy{shouldn't we be talking about $x$ here?}
(Fig.~\ref{fig:schematic}c).
The parameters of Equation~\ref{Eq-d} and $R_g$ were chosen so that the spatial and spectral focal curves intersect at 
the center of the O\,\textsc{v} \roy{\OV} image on the \CCD.

Starting from the analytically derived optical prescription, a model of the system was developed in ray-trace \roy{raytrace} software.
Since the instrument is radially symmetric, only one grating and its associated lightpath was analyzed. \roy{delete previous sentence, all lightpaths were analyzed}
In the ray trace model, $R_g$, $d_1$, $d_2$, grating cant angle, \CCD\ cant angle, and focus position were then 
optimized to minimize the RMS spot at select positions in the O\,\textsc{v} \roy{\OV} \FOV, illustrated in Fig.~\ref{fig:psf}.
The optical prescription derived from the ray trace is listed in Table~\ref{table:prescription} and 
Figure~\ref{fig:schematic}. """
            ))

            doc.append(figures.psf.figure())

            doc.append(figures.spot_size.figure())

            with doc.create(pylatex.Figure()) as figure:
                figure.add_image(str(figures.focus_curve.pdf()), width=None)
                figure.add_caption(pylatex.NoEscape(
                    r"""\roy{
Focus curve for the field angle at the middle of the \ESIS\ \FOV\ for the 
\defaultNumEmissionLines\ brightest wavelengths in the passband.
}"""
                ))
                figure.append(kgpy.latex.Label('fig:focusCurve'))

            with doc.create(pylatex.Figure()) as figure:
                figure.add_image(str(figures.vignetting.pdf()), width=None)
                figure.add_caption(pylatex.NoEscape(
                    r"""\roy{
(Top) 2D histogram counting the number of rays that were unvignetted by the \ESIS\ optical 
system as a function of field position.
The count is normalized to the maximum number of unvignetted rays at any field point.
The field and pupil grids have the same parameters as the grid for Figure~\ref{fig:spotSize}.
(Bottom) Residual between the top histogram and the vignetting model described in Table~\ref{table:vignetting}
}"""
                ))
                figure.append(kgpy.latex.Label('fig:vignetting'))

            doc.append(pylatex.NoEscape(r"""
\begin{equation}
\begin{split}
\left(x', y'\right) &= \C + \C_x x + \C_y y + \C_\lambda \lambda \\
&+ \C_{xx} x^2 + \C_{xy} x y + \C_{y \lambda} x \lambda \\
&+ \C_{yy} y^2 + \C_{y \lambda} y \lambda + \C_{\lambda \lambda} \lambda^2
\end{split}
\end{equation}
"""
            ))

            model_distortion = optics_single.rays_output.distortion.model()
            model_distortion_relative = optics_single.rays_output_relative.distortion.model()

            def fmt_coeff(coeff: u.Quantity):
                return kgpy.format.quantity(coeff.value * u.dimensionless_unscaled, scientific_notation=True, digits_after_decimal=2)

            x_max = 500 * u.pix
            y_max = 500 * u.pix
            lambda_max = optics_single.wavelength[..., 1] - optics_single.wavelength[..., 0]

            with doc.create(pylatex.Table()) as table:
                table._star_latex_name = True
                with table.create(pylatex.Center()) as centering:
                    with centering.create(pylatex.Tabular('ll|rr|rr')) as tabular:
                        tabular.escape = False
                        tabular.append('\multicolumn{2}{l}{Coefficient} & $x\'$ & $y\'$ & $x\'$ rel. & $y\'$ rel.\\\\')
                        # tabular.add_row(['Coefficient', '$x\'$', '$y\'$'])
                        tabular.add_hline()
                        for c, name in enumerate(model_distortion.x.coefficient_names):
                            tabular.add_row([
                                f'{name}',
                                f'({model_distortion.x.coefficients[c].unit:latex_inline})',
                                fmt_coeff(model_distortion.x.coefficients[c].squeeze()),
                                fmt_coeff(model_distortion.y.coefficients[c].squeeze()),
                                fmt_coeff(model_distortion_relative.x.coefficients[c].squeeze()),
                                fmt_coeff(model_distortion_relative.y.coefficients[c].squeeze()),
                                # f'{model_distortion.x.coefficients[c] * 500 * u.pix}',
                                # f'{model_distortion.x.coefficients[c].squeeze():0.3f}',
                                # f'{model_distortion.y.coefficients[c].squeeze():0.3f}',
                            ])

            with doc.create(pylatex.Figure()) as figure:
                figure.add_image(str(figures.distortion.pdf()), width=None)
                figure.add_caption(pylatex.NoEscape(
                    r"""\roy{
Plot of the magnified, undistorted field stop aperture vs. the distorted \OV\ image of the 
field stop aperture on the \ESIS\ detector.
The magnification factor used for the undistorted field stop aperture is the ratio of the grating exit arm to the 
grating entrance arm (\armRatio).
The distorted image of the field stop aperture was calculated using the \ESIS\ distortion model, described in 
Table~\ref{table:distortion}.
}"""
                ))
                figure.append(kgpy.latex.Label('fig:distortion'))

            with doc.create(pylatex.Figure()) as figure:
                figure._star_latex_name = True
                figure.add_image(str(figures.distortion_residual.pdf()), width=None)
                figure.add_caption(pylatex.NoEscape(
                    r"""\roy{
Magnitude of the residual between a linear distortion model and the raytrace model (top) and between a quadratic 
distortion model and the raytrace model (bottom). This figure demonstrates that a quadratic distortion model is
sufficient to achieve sub-pixel accuracy.
}"""
                ))
                figure.append(kgpy.latex.Label('fig:distortionResidual'))

            doc.append(pylatex.NoEscape(
                r"""The ray trace model was also used to quantify how mirror and positional tolerances affect the 
instrument's spatial resolution.
Each element of the model was individually perturbed, then a compensation applied to adjust the image on the \CCD.
The compensation optimized grating tip/tilt angle and \CCD\ focus position, so that the image was re-centered and RMS 
spot size minimized at the positions in Fig.~\ref{F-spot} \roy{minimized at the vertices of the field stop and the central field angle}.
We then computed the maximum change in RMS spot size over all spot positions between the optimized and perturbed models.
The computed positional tolerances for each element in the \ESIS\ optical system are listed in Table~\ref{table:tol}.

The imaging error budget is displayed in Table~\ref{table:tol}.
For the primary mirror and grating surface figure contributions, we choose the \MTF\ figures of merit from the surface 
roughness specifications described earlier.
To quantify the remaining entries, we assume that each term can be represented by a gaussian function of width 
$\sigma^2$ that ``blurs'' the final image.
The value of $\sigma$ then corresponds to the maximum change in RMS spot size for each term as it is perturbed in the 
tolerance analysis described above.
The value of the \MTF\ in the right-most column of Table~\ref{table:tol} is computed from 
each of the gaussian blur terms at the Nyquist frequency (\SI{0.5}{cycles\per arcsecond}).
From Table~\ref{table:tol}, we estimate the total \MTF\ of \ESIS\ to be $0.109$ at the Nyquist frequency.
Compared to, for example, the Rayleigh criterion of \SI{0.09}{cycles\per arcsecond}~\citep{Rayleigh_1879} we estimate 
the resolution of \ESIS\ to be essentially pixel limited.
Since \ESIS\ pixels span \SI{0.76}{\arcsecond} \roy{\plateScaleMean}, the resolution target in Table~\ref{table:scireq} is obtained by this 
design."""
            ))

            doc.append(pylatex.NoEscape(
                r"""
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
\end{table*}
"""
            ))

            opt = esis.optics.design.final(**optics.error_kwargs)
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
                    plt.plot(frequency.x.take(indices=0, axis=rays.axis.pupil_x).squeeze(), mtf.take(indices=0, axis=rays.axis.pupil_x).squeeze())
                    plt.plot(frequency.y.take(indices=0, axis=rays.axis.pupil_y).squeeze(), mtf.take(indices=0, axis=rays.axis.pupil_y).squeeze())

                mtf = np.take(a=mtf, indices=[0], axis=rays.axis.pupil_x)
                print('mtf', mtf.mean())
                index_frequency_requirement = np.argmax(frequency.y.take(indices=[0], axis=rays.axis.pupil_x) >= frequency_requirement, axis=rays.axis.pupil_y)
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

            distance_grating_to_detector = (optics_single.detector.transform.translation_eff - optics_single.grating.transform.translation_eff).length
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

            with doc.create(pylatex.Table()) as table:
                table._star_latex_name = True
                with table.create(pylatex.Center()) as centering:
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
                            psf_size=ptp_to_rms(2 * np.sin(opt.sparcs.rlg_drift * opt.detector.exposure_length / 2) * pointing / opt.plate_scale.x),
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
                table.add_caption(pylatex.NoEscape(
                    f"""
Imaging error budget and tolerance analysis results. \\MTF\\ is given at
{kgpy.format.quantity(frequency_mtf_arcsec, digits_after_decimal=1)}."""
                ))
                table.append(kgpy.latex.Label('table:errorBudget'))

        with doc.create(pylatex.Subsection('Vignetting')):
            doc.append(pylatex.NoEscape(r"""
The original design of \ESIS\ had no vignetting thanks to an stop placed at the primary mirror that was designed to 
perfectly fill the grating with the same amount of light for each point in the \FOV.
This is the \ESIS\ design that was used for the optimization procedure of the grating parameters described in 
Section~\ref{subsec:OptimizationandTolerancing}, for example.
All other results described in the paper use the fully-open system.
Before flight, we decided to remove the primary aperture stop to increase the sensitivity of the instrument at the
expense of introducing vignetting to the \ESIS\ \FOV.
This was acceptable since the vignetting was found to be a simple linear field as shown in Figure~\ref{fig:vignetting},
and could be removed in the post-processing phase.
"""
            ))

        with doc.create(pylatex.Subsection('Distortion')):
            doc.append(pylatex.NoEscape(
                r"""
The distortion is due to two factors: first, the tilt of the detector as needed to maintain good focus over the \FOV 
\citep{Poletto04}; second, the anamorphic magnification of the grating (see \cite{Schweizer1979}).
"""
            ))

        with doc.create(pylatex.Subsection('Coatings and Filters')):

            with doc.create(pylatex.Figure()) as figure:
                figure.add_image(str(figures.grating_multilayer_schematic.pdf()), width=None)
                figure.add_caption(pylatex.NoEscape(
                    r"""
Schematic of the Al/SiC/Mg \roy{\gratingCoatingMaterialShort} multilayer with $N=4$ \roy{$N=\gratingCoatingNumLayers$} layers.
"""
                ))
                figure.append(kgpy.latex.Label('fig:gratingMultilayerSchematic'))

            with doc.create(pylatex.Figure()) as figure:
                figure.add_image(str(figures.grating_efficiency_vs_angle.pdf()), width=None)
                figure.add_caption(pylatex.NoEscape(
                    r"""
Measured efficiency \roy{at \gratingTestWavelength} of a single grating \roy{the Channel \testGratingChannelIndex\ grating} as a function of reflection angle on \roy{\testGratingDate}.
Note flat response in first order over instrument \FOV\ and suppression of zero order.
"""
                ))
                figure.append(kgpy.latex.Label('fig:gratingEfficiencyVsAngle'))

            with doc.create(pylatex.Figure()) as figure:
                figure.add_image(str(figures.component_efficiency_vs_wavelength.pdf()), width=None)
                figure.add_caption(pylatex.NoEscape(
                    r"""(Top) Measured reflectance for several multilayer coated witness samples 
\roy{at an incidence angle of \gratingWitnessMeasurementIncidenceAngle\ on \testGratingDate.
The white regions indicate wavelengths that intercept the detector and the gray regions indicate wavelengths that
miss the detector.
Note the suppression of second order relative to the first order and the consistency of the coatings between each 
channel.
The Channel \gratingWitnessMissingChannel\ grating measurement is missing due to issues in the measurement apparatus.
(Bottom) Comparison of the efficiency of the three main \ESIS\ optical components: primary mirror, grating and filter.
The primary mirror efficiency is based on measurements of a \Si\ witness sample taken on \primaryMeasurementDate\ at an 
angle of incidence of \primaryWitnessMeasurementIncidenceAngle. 
The grating efficiency is from a measurement of the Channel \testGratingChannelIndex\ grating taken on \testGratingDate\
at an angle of incidence of \gratingMeasurementIncidenceAngle.
The filter efficiency is a theoretical model that includes the filter mesh, \filterThickness\ of \filterMaterial\ and
\filterOxideThickness\ of \filterMaterial\ oxide.
}"""
                ))
                figure.append(kgpy.latex.Label('fig:componentEfficiencyVsWavelength'))

            with doc.create(pylatex.Figure()) as figure:
                figure._star_latex_name = True
                figure.add_image(str(figures.grating_efficiency_vs_position.pdf()), width=None)
                figure.add_caption(pylatex.NoEscape(
                    r"""
\roy{Channel \testGratingChannelIndex\ grating efficiency at \gratingTestWavelength\ vs. position for two orthogonal slices across the optical 
surface on \testGratingDate.}"""
                ))
                figure.append(kgpy.latex.Label('fig:gratingEfficiencyVsPosition'))

            doc.append(pylatex.NoEscape(
                r"""
The diffraction gratings are coated with a multilayer optimized for a center wavelength of \SI{63.0}{\nano\meter} \roy{\OV}, 
developed by a collaboration between Reflective X-Ray Optics LLC and \LBNL.
In Fig.~\ref{fig:gratingEfficiencyVsAngle}, characterization of a single, randomly selected multilayer coated grating at \LBNL\ shows 
that the grating reflectivity is constant over the instrument \FOV\ in the $m=1$ order while the $m=0$ order is almost 
completely suppressed.
Figure~\ref{fig:gratingMultilayerSchematic} shows a schematic of the coating that achieves peak reflectivity and selectivity in the 
$m=0$ order using four \roy{\gratingCoatingNumLayersWords} layer pairs of silicon carbide (SiC) \roy{\firstGratingCoatingMaterial} and magnesium (Mg) \roy{\secondGratingCoatingMaterial}.
The Aluminum (Al) \roy{\thirdGratingCoatingMaterial} layers are deposited adjacent to each Mg \roy{\firstGratingCoatingMaterial} layer to mitigate corrosion.
\roy{As Charles mentioned, this doesn't make sense. Why can it go \firstGratingCoatingMaterial\ to \secondGratingCoatingMaterial, but not \secondGratingCoatingMaterial\ to \firstGratingCoatingMaterial?}

The maximum reflectance for the coating alone in the nominal instrument passband is $\sim$\SI{35}{\percent} \roy{\gratingWitnessEfficiency} in 
the upper panel of Figure~\ref{fig:componentEfficiencyVsWavelength}, measured from witness samples coated at the same time as the diffraction gratings.
Combined with the predicted groove efficiency from \S\,\ref{subsec:Optics} and, given the relatively shallow groove profile 
and near normal incidence angle, the total reflectivity in first order is $\sim$\SI{13}{\percent} \roy{\gratingEfficiency} at 
\SI{63}{\nano\meter} \roy{\OV}.
This is confirmed by the first order efficiency measured from a single \ESIS\ grating in the lower panel of 
Figure~\ref{fig:componentEfficiencyVsWavelength}.  

Unlike \EUV\ imagers (\eg, \TRACE~\citep{Handy99}, \AIA~\citep{Lemen12}, and the \HiC~\citep{Kobayashi2014}) 
the \ESIS\ passband is defined by a combination of the field stop and grating (\S\,\ref{subsec:Optics}, 
Fig.~\ref{fig:projections}) rather than multi-layer \roy{multilayer} coatings.
The coating selectivity is therefore not critical in this respect, allowing the multi-layer \roy{multilayer} to be manipulated to 
suppress out-of-band bright, nearby emission lines.
The lower panel of Figure~\ref{fig:componentEfficiencyVsWavelength} shows the peak reflectance of the grating multilayer is shifted slightly towards longer 
wavelengths to attenuate the He\,\textsc{i} \roy{\HeI} emission line, reducing the likelihood of detector saturation.
A similar issue arises with the bright He\,\textsc{ii} (\SI{30.4}{\nano\meter}) \roy{\HeII} line.
Through careful design of the grating multilayer, the reflectivity at this wavelength is $\sim$\SI{2}{\percent} \roy{\gratingHeIIRejectionRatio} of that 
at \SI{63}{\nano\meter} \roy{\OV} (lower panel of Figure~\ref{fig:componentEfficiencyVsWavelength}).
In combination with the primary mirror coating (described below) the rejection ratio at \SI{30.4}{\nano\meter} \roy{\HeIIwavelength} is 
$\sim$\SI{32}{\decibel} \roy{\totalHeIIRejection}.  Thus, He\,\textsc{ii} \roy{\HeII} emission will be completely attenuated at the \CCD.

The flight and spare primary mirrors were coated with the same Al/SiC/Mg \roy{\gratingCoatingMaterialShort} multilayer.
Corrosion of this multilayer rendered both mirrors unusable.
The failed coating was stripped from primary mirror SN001.
The mirror was then re-coated with a \SI{5}{\nano\meter} \roy{\primaryCoatingBaseThickness} thick layer of chromium (Cr) \roy{\primaryCoatingBaseMaterial} to improve adhesion followed by a 
\SI{25}{\nano\meter} \roy{\primaryCoatingThickness} thick layer of SiC \roy{\primaryCoatingMaterial}.
The reflectance of this coating deposited on a \Si\ wafer witness sample appears in 
Fig.~\ref{fig:componentEfficiencyVsWavelength}.
The spare primary mirror (SN002) retains the corroded Al/SiC/Mg \roy{\gratingCoatingMaterialShort} multilayer.

The \Si\ \CCDs\ are sensitive to visible light as well as \EUV.
Visible solar radiation is much stronger than \EUV, and visible stray light can survive multiple scatterings while 
retaining enough intensity to contaminate the \EUV\ images.
Lux\'el \citep{Powell90} Al \roy{\filterMaterial} filters \SI{100}{\nano\meter} \roy{\filterThickness} thick will be 
used to shield each \CCD\ from visible light.
The Al \roy{\filterMaterial} film is supported by a 70 line per inch (lpi) \roy{\filterMeshPitch} Ni \roy{\filterMeshMaterial} mesh, with 82\% \roy{\filterMeshRatio} transmission.
The theoretical filter transmission curve, modeled from CXRO data \citep{Henke93}, is displayed in 
Fig.~\ref{fig:componentEfficiencyVsWavelength}.
We conservatively estimate filter oxidation at the time of launch as a 4nm \roy{\filterOxideThickness} thick layer of Al$_2$O$_3$.

An Al \roy{\filterMaterial} filter is positioned in front of the focal plane of each \CCD\ by a filter tube, creating a light tight \roy{light-tight} box with a 
labyrinthine evacuation vent (e.g., Fig.~\ref{F-cameras}).
The placement of the filter relative to the \CCD\ is optimized so that the filter mesh shadow is not visible.
By modeling the filter mesh shadow, we find that a position far from the \CCD\ ($>$\SI{200}{\milli\meter} \roy{\filterToDetectorDistance}) and mesh grid
clocking of \SI{45}{\degree} \roy{\filterClocking} to the detector array reduces the shadow amplitude well below photon statistics.
The \MOSES\ instrument utilizes a similar design;
no detectable signature of the filter mesh is found in data and inversion residuals from the 2006 \MOSES\ flight.

To prevent oxidation, and to minimize the risk of tears, pinholes, and breakage from handling, the filters will be 
stored in a nitrogen purged environment until after payload vibration testing."""
            ))

        with doc.create(pylatex.Subsection('Sensitivity and Cadence')):
            doc.append(pylatex.NoEscape(
                r"""

Count rates for \ESIS\ are estimated using the expected component throughput from Section~\ref{
subsec:CoatingsandFilters} and the \CCD\ \QE\ listed in Table~\ref{table:prescription}. Line intensities are derived 
from \citet{Vernazza78} (V\&R) \roy{\VR} and the \SOHO/\CDS\ \citep{Harrison95} data, and are given in a variety of 
solar contexts: \QS, \CHs, and \ARs. The \SI{100}{\percent} duty cycle of \ESIS\ (\S\,\ref{subsec:Cameras}) gives us 
the flexibility to use the shortest exposures that are scientifically useful. So long as the shot noise dominates 
over read noise (which is true even for our coronal hole estimates at \SI{10}{\second} exposure length), we can stack 
exposures without a significant \SNR\ penalty. Table~\ref{table:count} shows that \ESIS\ is effectively shot noise 
limited with a \SI{10}{\second} exposure. The signal requirement in Table~\ref{table:scireq} is met by stacking 
exposures. Good quality images ($\sim300$ counts) in active regions can be obtained by stacking \SI{30}{\second} 
worth of exposures. This cadence is sufficient to observe explosive events, but will not resolve torsional Alfv\'en 
waves described in \S\,\ref{sec:ScienceObjectives}. However, by stacking multiple \SI{10}{\second} exposures, 
sufficient \SNR\ \emph{and} temporal resolution of torsional Alfv\'en wave oscillations can be obtained. \roy{Just 
delete these next three sentences?} \jake{Assuming the table and sentences above have been updated to reflect the 
vignetted system, yes}. We also note that the count rates given here are for an unvignetted system which is limited 
by the baffling of this design. While not explored here, there is the possibility of modifying the instrument 
baffling (\S\,\ref{subsec:AperturesandBaffles}) to increase throughput. Thus, a faster exposure cadence may be 
obtained by accepting some vignetting in the system. 

\begin{table}
    \centering
    \begin{tabular}{lcccc}
        Source & V\&R & V\&R & V\&R & CDS \\
        Solar Context & QS & CH & AR & AR \\
        \hline
        \multicolumn{5}{c}{\SI{10}{\second} Exp.}\\
        Mg\,\textsc{x} (\SI{62.5}{\nano\meter}) & 3 & 0 & 26 & 16  \\
        O\,\textsc{V} (\SI{62.9}{\nano\meter}) & 22 & 19 & 66 & 34 \\
        \hline
        Total Counts & 25 & 19 & 92 & 50 \\
        Shot Noise & 5.0 & 4.3 & 9.6 & 7.0 \\
        Read Noise (est.) & \multicolumn{4}{c}{-- 1.9 --} \\
        SNR & 4.7 & 4.0 & 9.4 & 6.8 \\
        \hline \hline
        \multicolumn{5}{c}{$3\times$\SI{10}{\second} Exp. Stack}\\
        Total Counts & 75 & 56 & 276 & 148 \\
        SNR & 8.1 & 6.8 & 16.3 & 11.7 \\
        \hline
    \end{tabular}
    \caption{
        Estimated signal statistics per channel (in photon counts) for \ESIS\ lines in coronal hole (CH), quiet Sun (QS), and active region (AR).
    }
    \label{table:count}
\end{table}
"""
            ))

        intensity_o5 = [334.97, 285.77, 1018.65, 519.534] * u.erg / u.cm ** 2 / u.sr / u.s
        intensity_mg10 = [51.43, 2.62, 397.64, 239.249] * u.erg / u.cm ** 2 / u.sr / u.s

        energy_o5 = wavelength_o5.to(u.erg, equivalencies=u.spectral()) / u.photon
        energy_mg10 = wavelength_mg10_2.to(u.erg, equivalencies=u.spectral()) / u.photon

        optics_single_measured = optics.as_measured_single_channel()
        rays = optics_single_measured.rays_output

        area = rays.intensity.copy()
        area[~rays.mask] = np.nan
        area = np.nansum(area, (rays.axis.pupil_x, rays.axis.pupil_y, rays.axis.velocity_los), keepdims=True)
        area[area == 0] = np.nan
        area = np.nanmean(area, (rays.axis.field_x, rays.axis.field_y)).squeeze()
        area_o5 = area[0]
        area_mg10 = area[2]

        pixel_subtent = (optics_single.plate_scale.x * optics_single.plate_scale.y * u.pix * u.pix).to(u.sr)
        time_integration = optics_single.detector.exposure_length

        counts_o5 = (intensity_o5 * area_o5 * pixel_subtent * time_integration / energy_o5).to(u.photon)
        counts_mg10 = (intensity_mg10 * area_mg10 * pixel_subtent * time_integration / energy_mg10).to(u.photon)
        counts_total = counts_o5 + counts_mg10

        stack_num = 12
        counts_total_stacked = counts_total * stack_num

        noise_shot = np.sqrt(counts_total.value) * counts_total.unit
        noise_shot_stacked = np.sqrt(counts_total_stacked.value) * counts_total.unit

        noise_read = optics_single_measured.detector.readout_noise.mean()
        noise_read = noise_read * optics_single_measured.detector.gain.mean()
        noise_read_o5 = (noise_read / (energy_o5 / (3.6 * u.eV / u.electron))).to(u.photon)
        noise_read_o5_stacked = stack_num * noise_read_o5

        noise_total = np.sqrt(np.square(noise_shot) + np.square(noise_read_o5))
        noise_total_stacked = np.sqrt(np.square(noise_shot_stacked) + np.square(noise_read_o5_stacked))

        snr = counts_total / noise_total
        snr_stacked = counts_total_stacked / noise_total_stacked

        label = f'1 $\\times$ {kgpy.format.quantity(time_integration, digits_after_decimal=0)} exp.'
        label_stacked = f'{stack_num} $\\times$ {kgpy.format.quantity(time_integration, digits_after_decimal=0)} exp.'

        doc.set_variable(
            name='NumExpInStack',
            value=str(stack_num),
        )

        doc.set_variable_quantity(
            name='StackedCoronalHoleSNR',
            value=snr_stacked[np.argmin(intensity_o5)],
            digits_after_decimal=1,
        )

        with doc.create(pylatex.Table()) as table:
            # table._star_latex_name = True
            with table.create(pylatex.Center()) as centering:
                with centering.create(pylatex.Tabular('lrrrr')) as tabular:
                    tabular.escape = False
                    tabular.add_row([r'Source', r'\VR', r'\VR', r'\VR', r'\CDS'])
                    tabular.add_row(r'Solar context', r'\QSShort', r'\CHShort', r'\ARShort', r'\ARShort')
                    tabular.add_hline()
                    tabular.add_hline()
                    tabular.append(f'\\multicolumn{{5}}{{c}}{{{label}}}\\\\')
                    tabular.add_row([r'\OV', ] + [f'{c:0.0f}' for c in counts_o5.value])
                    tabular.add_row([r'\MgXdim',] + [f'{c:0.0f}' for c in counts_mg10.value])
                    tabular.add_hline()
                    tabular.add_row([r'Total', ] + [f'{c:0.0f}' for c in counts_total.value])
                    tabular.add_row([r'Shot noise', ] + [f'{c:0.1f}' for c in noise_shot.value])
                    tabular.add_row([r'Read noise', ] + 4 * [f'{noise_read_o5.value:0.1f}'])
                    tabular.add_row([r'\SNRShort', ] + [f'{c:0.1f}' for c in snr.value])
                    tabular.add_hline()
                    tabular.add_hline()
                    tabular.append(f'\\multicolumn{{5}}{{c}}{{{label_stacked}}}\\\\')
                    tabular.add_row(['Total', ] + [f'{c:0.0f}' for c in counts_total_stacked.value])
                    tabular.add_row([r'\SNRShort', ] + [f'{c:0.1f}' for c in snr_stacked.value])
                    tabular.add_hline()
                    tabular.add_hline()

            table.add_caption(pylatex.NoEscape(
                r"""
Estimated signal statistics per channel (in photon counts) for \ESIS\ lines in \CH, \QS, and \AR."""
            ))
            table.append(kgpy.latex.Label('table:counts'))

        with doc.create(pylatex.Subsection('Alignment and Focus')):
            doc.append(pylatex.NoEscape(
                r"""
In the conceptual phase of \ESIS, the decision was made to perform focus and alignment in visible light with a \HeNe\ 
source.
Certain difficulties are introduced by this choice, however, the benefits outweigh the operational complexity and 
equipment that would be required for focus in \EUV.
Moreover, a sounding rocket instrument requires robust, adjustment-free mounts to survive the launch environment.
Such a design is not amenable to iterative adjustment in vacuum.  The choice of alignment wavelength is arbitrary for 
most components;
\CCD\ response and multilayer coating reflectively is sufficient across a wide band a visible wavelengths.
The exceptions are the thin film filters (which will not be installed until just before launch and have no effect on 
telescope alignment and focus) and the diffraction gratings.
Visible light gratings have been manufactured specifically for alignment and focus.
These gratings are identical to the \EUV\ flight version, but with a ruling pattern scaled to a 
\SI{632.8}{\nano\meter} wavelength.
"""
            ))

            with doc.create(pylatex.Figure()) as figure:
                figure.add_image('figures/old/Alignment_transfer_1gr_text', width=kgpy.latex.columnwidth)
                figure.add_caption(pylatex.NoEscape(
                    r"""
\jake{Will update this figure.  Will include a rendering of the secondary mount pointing to the tuffet, grating
backplate, bipod, etc. Capturing the same as before but without TEA.}
\ESIS\ alignment transfer device, consisting of three miniature confocal microscopes that translate along the optical 
axis.  
Trapezoidal grating, bipods, and mounting plate are installed on the tuffet in front of the apparatus 
(left of center)"""
                ))
                figure.append(kgpy.latex.Label('F-alt'))

            doc.append(pylatex.NoEscape(
                r"""
After alignment and focus has been obtained with the \HeNe\ source, the instrument will be prepared for flight by 
replacing the visible gratings by the \EUV\ flight versions.
Each grating (\EUV\ and alignment) is individually mounted to a backing plate using a bipod system similar to that of the 
primary mirror.
The array of gratings on their backing plates are in turn mounted to a single part which we call the `tuffet.'
The backing plate is attached to the tuffet by bolts through spherical washers.
With this mounting scheme, the gratings can be individually aligned in tip/tilt.
The tuffet is attached to the secondary mirror mount structure (Fig.~\ref{F-alt}).
This enables the entire grating array to be replaced simply by removing/installing the tuffet, such as when switching 
between alignment and \EUV\ gratings.

Properly positioning the gratings will be the most difficult aspect of final telescope assembly.
Table~\ref{table:tol} shows that the telescope is very sensitive to defocus.
The depth of field is an order of magnitude smaller in \EUV\ than in visible light.
Moreover, the sensitivity to defocus at the gratings is $M^2+1=17$ times greater than at the detectors.
Another sensitive aspect of the telescope is grating tip/tilt.
A tolerance of $\sim\pm$\SI{0.5}{\milli\radian} will be needed to ensure that the entire image lands on the active area 
of the \CCD.

%Once the visible gratings are aligned and focused, the challenge is to transfer this alignment to the UV gratings.
%\citet{Johnson18} describes a procedure and the apparatus constructed to accurately transfer the position of an 
%alignment grating radius to an \EUV\ flight grating.
%This device, displayed in Fig.~\ref{F-alt}, consists of an array of three miniature confocal microscopes that record the 
%position of the alignment grating radius.
%The alignment grating is replaced by an \EUV\ grating, which is then measured into position by the same apparatus.
%This device (and procedure) is capable of obtaining position measurements of the diffraction gratings to a repeatability 
%of $\approx$\SI{14}{\micro\meter} in the three confocal channels.
%This alignment transfer apparatus will ensure that the \EUV\ flight gratings are positioned to within the tolerances 
%described in Table~\ref{table:tol}.

Once the visible gratings were aligned and focused, the challenge is to transfer this alignment to the \EUV\ flight 
gratings.
We performed this transfer using a 4D Phasecam 6000 interferometer.
We aligned the interferometer to each visible grating such that the tip, tilt, and defocus aberrations were zero.
Then, placing the corresponding flight grating in the visible grating's place, we shimmed the mounting screws of the 
flight grating to match the tip, tilt, and defocus of the alignment grating.
Since the mounting interface for the tuffet is extremely repeatable by design we were able to swap tuffets and compare 
their alignment and focus to ensure everything was transferred correctly.

When transferring the focus and alignment two key details were considered.
First, there is nothing about the tuffet that constrains the grating roll.
Therefore, we needed to ensure each flight grating had the same roll as each visible grating.
This was accomplished by using a \HeNe\ laser diverged through a cylindrical optic that illuminated each grating with a 
line perpendicular to the grating blaze direction.
The line of light was reflected back onto a ruled target that could be compared between gratings.
Since our alignment gratings were ruled to image light of approximately an order of magnitude longer wavelength the
laser and cylindrical optic were placed at the position of Littrow for the 10th order image in the visible grating,
and the 1st order for the \EUV\ gratings.
Second, during testing we measured slight differences in radius of curvature between each grating. 
Therefore each flight grating was prescribed a specific amount of defocus to account for the difference in radius of
curvature between each optic when transferring alignment and focus.
"""
            ))

        with doc.create(pylatex.Subsection('Apertures and Baffles')):

            with doc.create(pylatex.Figure()) as figure:
                figure._star_latex_name = True
                figure.add_image('figures/old/Baffles_1clr', width=kgpy.latex.textwidth)
                figure.add_caption(pylatex.NoEscape(
                    r"""Model view of \ESIS\ baffle placement and cutouts."""
                ))
                figure.append(kgpy.latex.Label('F-Baff1'))

            doc.append(pylatex.NoEscape(
                r"""
\jake{Needs a big rework}               
Each channel of \ESIS\ has two apertures: one at the surface of the grating and another in front of the primary mirror.
The purpose of the aperture at the grating is to mask the out-of-figure margins at the edges of these optics.
This provides a well defined edge to the clear aperture of each grating while also keeping unwanted rays from being 
reflected from the grating margins and back onto the \CCDs.
The dimensions of the grating aperture match those of the grating clear aperture shown in Figure~\ref{fig:schematic}c. 

The aperture placed at the primary mirror is the stop for each individual channel.
The area of the stop has been maximized under the constraint that no rays be vignetted anywhere else in the system.
The gratings and their clear apertures were the most significant areas of concern for potential vignetting.
Thus, the shape of stop at the primary is largely influenced by the shape of the grating clear aperture.
The inner extent of the primary stop (the ``tip'' of the triangle in Figure~\ref{fig:schematic}b) is defined by the 
occultation of the primary by the shadow cast from the gratings and their mounts.
This presented an intricate geometry problem, as the occultation is a function of the incoming field angle, 
the radial extent of the grating mount, and the distance of the mount to the primary mirror along the optical axis.
Hence, the inner extent of the primary stop was solved for iteratively with the optimization described in 
Section~\ref{subsec:OptimizationandTolerancing}, which affected the placement of the gratings relative to the primary mirror.
The resulting optimized and non-vignetting stop geometry is shown in Figure~\ref{fig:schematic}b.

After final optimization, the stop geometry was analyzed to check for vignetting at the grating with the optical model.
A footprint diagram was generated at the grating from of multiple grids of rays.
The incidence angle of each grid of rays corresponded to the extremes of \FOV\ defined by the positions of the eight 
points of the octagonal field stop.
The footprint diagram showed that, with the stop completely filled, no ray landed outside of the grating clear aperture
in Figure~\ref{fig:schematic}c, and no ray was intercepted by the central obscuration.

From Figure~\ref{fig:schematic}c it is apparent that considerable surface area of the primary mirror is unused by the 
non-vignetting stop design.
The primary apertures could be enlarged considerably if the vignetting constraint were to be relaxed.

The \ESIS\ baffles are designed to block direct light paths between the front aperture plate and the \CCDs\ for any ray 
$<$\SI{1.4}{\degree} from the optical axis.
This angle is purposefully larger than the angular diameter of the sun ($\sim$\SI{0.5}{\degree}) so that any direct 
paths are excluded from bright sources in the solar corona.
All baffles are bead-blasted, anodized \Al\ sheet metal oriented perpendicular to the optical axis.
The size and shape of the cutouts were determined using a combination of the ray trace from 
Section~\ref{subsec:OptimizationandTolerancing} and 3D modeling.
The light path from the primary mirror to the field stop is defined as the volume that connects each vertex of the 
primary mirror aperture mask (e.g., Fig.~\ref{fig:schematic}) to every vertex of the octagonal field stop.
This is a conservative definition that ensures no rays within the \FOV\ are excluded, and therefore unintentionally 
vignetted by the baffles.  Light paths from the field stop to the grating, and from the grating to the image formed on 
the \CCD, are defined in a similar manner.
The cutouts in the baffles are sized using the projection of these light paths onto the baffle surface.
A conservative \SI{1}{\milli\meter} margin is added to each cutout to prevent unintentional vignetting.
A model of the six baffles, showing cutouts and position on the optical bench, is displayed in Fig.~\ref{F-Baff1}."""
            ))

        with doc.create(pylatex.Subsection('Cameras')):

            with doc.create(pylatex.Figure()) as figure:
                figure.add_image('figures/old/ESIS_Cameras_1gr_text', width=kgpy.latex.columnwidth)
                figure.add_caption(pylatex.NoEscape(
                    r"""
\ESIS\ camera assembly as built by \MSFCShort.  
Thin film filters and filter tubes are not installed in this image."""
                ))
                figure.append(kgpy.latex.Label('F-cameras'))

            doc.append(pylatex.NoEscape(
                r"""
The \ESIS\ \CCD\ cameras were designed and constructed by \MSFC\ and are the latest in a 
series of camera systems developed specifically for use on solar space flight instruments.
The \ESIS\ camera heritage includes those flown on both the \CLASP~\citep{Kano12,Kobayashi12} and \HiC~\citep{Kobayashi2014}.

The \ESIS\ detectors are CCD230-42 \roy{\detectorName} astro-process \CCDs\ from E2V \roy{\detectorManufacturer}.
For each camera, the \CCD\ is operated in a split frame transfer mode with each of the four ports read out by a 16-bit \roy{\detectorAnalogToDigitalBits-bit} A/D 
converter.
The central $2048 \times 1024$ \roy{$\detectorPixelsX \times \detectorPixelsY$} pixels of the $2k\times2k$ device are used for imaging, while the outer two regions are 
used for storage.
Two \roy{\DetectorNumOverscanColumnWords} overscan columns on either side of the imaging area and eight extra rows in each storage region will monitor read 
noise and dark current.
When the camera receives the trigger signal, it transfers the image from the imaging region to the storage regions and 
starts image readout.
The digitized data are sent to the \DACS\ through a SpaceWire interface immediately, 
one line at a time.
The frame transfer takes $<$\SI{60}{\milli\second} \roy{\detectorFrameTransferTime}, and readout takes \SI{1.1}{\second} \roy{\detectorReadoutTime}.
The cadence is adjustable from 2-\SI{600}{\second} \roy{\detectorExposureLengthRange} in increments of \SI{100}{\milli\second} \roy{\detectorExposureLengthIncrement}, to satisfy the requirement 
listed in Table~\ref{table:scireq}.
Because the imaging region is continuously illuminated, the action of frame transfer (transferring the image from the 
imaging region to the storage regions) also starts the next exposure without delay.
Thus the exposure time is controlled by the time period between triggers.
Camera 1 \roy{\detectorTriggerIndex} (Fig.~\ref{F-cameras}) generates the sync trigger, which is fed back into Camera 1's \roy{\detectorTriggerIndex's} trigger input and provides 
independently buffered triggers to the remaining three cameras.
The trigger signals are synchronized to better than $\pm$\SI{1}{\milli\second} \roy{$\pm$\detectorSynchronizationError}.
Shutterless operation allows \ESIS\ to observe with a \SI{100}{\percent} duty cycle.
The cadence is limited only by the 1.1\,s \roy{\detectorReadoutTime} readout time. 

\MSFC\ custom designed the camera board, enclosure, and mounting structure for \ESIS\ to fit the unique packaging 
requirements of this experiment (Fig~\ref{F-cameras}).
The front part of the camera is a metal block which equalizes the temperature across the \CCD\ while fastening it in 
place.
The carriers of all cameras are connected to a central two-piece copper (\SI{3}{\kilo\gram}) and aluminum 
(\SI{1}{\kilo\gram}) thermal reservoir (cold block) by flexible copper cold straps.
The flexible cold straps allow individual cameras to be translated parallel to the optical axis (by means of shims) up 
to $\sim$\SI{13}{\milli\meter} \roy{$\sim$\detectorFocusAdjustmentRange} to adjust focus in each channel prior to launch.
The centrally located cold block will be cooled by LN2 \roy{\LN} flow from outside the payload until just before launch.
The LN2 \roy{\LN} flow will be controlled automatically by a Ground Support Equipment (GSE) \roy{\GSE} computer so that all cameras are 
maintained above survival temperature but below the target temperature of \SI{-55}{\celsius} \roy{\detectorTemperatureTarget} to insure a negligible dark 
current level.

The gain, read noise, and dark current of the four cameras were measured at \MSFC\ using an ${}^{55}$Fe radioactive 
source.
Cameras are labeled 1, 2, 3, and 4 \roy{\channelNames} with associated serial numbers SN6, SN7, SN9, and SN10 \roy{\detectorSerialNumbers} respectively in 
Fig.~\ref{F-cameras}.  Gain ranges from 2.5-\SI{2.6}{e^- \per DN} \roy{\detectorGainRange} in each quadrant of all four cameras.
Table~\ref{T-cameras} lists gain, read noise, and dark current by quadrant for each camera.  

The \QE\ of the \ESIS\ \CCDs\ will not be measured before flight.
Similar astro-process \CCDs\ with no AR \roy{antireflection (because AR is already used for active region)} coating are used in the \SXI\ aboard the \GOES\ N and O.
A \QE\ range of 43\% at 583\AA\ \roy{\detectorQuantumEfficiencyHeI\ at \HeI} to 33\% at 630\AA\ \roy{\detectorQuantumEfficiency\ at \OV} is expected for the \ESIS\ \CCDs, based on \QE\ measurements by 
\citet{Stern04} for \GOES\ \SXI\ instruments.

\begin{table}[!htb]
\caption{\ESIS\ Camera properties.}
%\tableformat
\begin{tabular}{ccccc}
Camera & Quad & Gain & Read Noise & Dark Current \\
 & & [$e^-/DN$] & [DN] & [$e^-/ms$] \\
\hline %-----------------------------------------------------------------------------
1 (SN6) & 1 & 2.57 & 3.9 & $1.37e^-4$ \\
& 2 & 2.50 & 4.0 & $9.66e^-5$ \\
& 3 & 2.52 & 4.1 & $6.85e^-5$ \\
& 4 & 2.53 & 3.7 & $9.80e^-5$ \\ 
\hline
2 (SN7) & 1 & 2.55 & 3.9 & $6.77e^-5$ \\
& 2 & 2.58 & 4.0 & $5.89e^-5$ \\
& 3 & 2.57 & 4.0 & $8.98e^-5$ \\
& 4 & 2.63 & 4.0 & $1.01e^-4$ \\ 
\hline %-----------------------------------------------------------------------------
3 (SN9) & 1 & 2.57 & 4.1 & $3.14e^-5$ \\
& 2 & 2.53 & 4.1 & $2.68e^-5$ \\
& 3 & 2.52 & 4.1 & $3.18e^-5$ \\
& 4 & 2.59 & 4.3 & $3.72e^-5$ \\ 
\hline
4 (SN10) & 1 & 2.60 & 3.9 & $6.39e^-4$ \\
& 2 & 2.60 & 3.9 & $5.07e^-5$ \\
& 3 & 2.54 & 4.2 & $6.63e^-5$ \\
& 4 & 2.58 & 4.1 & $8.24e^-5$ \\ 
\hline
\end{tabular}
\label{T-cameras}
\end{table}"""
            ))

            detector = optics_all.detector

            with doc.create(pylatex.Table()) as table:
                # table._star_latex_name = True
                with table.create(pylatex.Center()) as centering:
                    with centering.create(pylatex.Tabular('ccccc')) as tabular:
                        tabular.escape = False
                        tabular.add_row([r'Channel', r'Quad.', r'Gain', r'Read noise', r'Dark current',])
                        tabular.add_row(['', '', f'({detector.gain.unit:latex_inline})', f'({detector.readout_noise.unit:latex_inline})', f'({detector.dark_current.unit:latex_inline})'])
                        tabular.add_hline()
                        for i in range(detector.gain.shape[0]):
                            for j in range(detector.gain.shape[1]):
                                if j == 0:
                                    channel_name_i = optics_all.channel_name[i]
                                    serial_number_i = f'({detector.serial_number[i]})'
                                else:
                                    channel_name_i = ''
                                    serial_number_i = ''
                                tabular.add_row([
                                    f'{channel_name_i} {serial_number_i}',
                                    j + 1,
                                    detector.gain[i, j].value,
                                    detector.readout_noise[i, j].value,
                                    f'{detector.dark_current[i, j].value:0.3f}',
                                ])
                            tabular.add_hline()
                table.add_caption(pylatex.NoEscape(r'\ESIS\ camera properties'))
                table.append(kgpy.latex.Label('tabel:cameraProperties'))

        with doc.create(pylatex.Subsection('Avionics')):
            doc.append(pylatex.NoEscape(
                r"""
The \ESIS\ \DACS\ are based on the designs used for both \CLASP~\citep{Kano12,Kobayashi12} and \HiC~\citep{Kobayashi2014}.
The electronics are a combination of \MOTS\ hardware and custom designed components.
The \DACS\ is a 6-slot, 3U, open VPX PCIe architecture conduction cooled system using an AiTech C873 single board
computer.
The data system also include a \MOTS\ PCIe switch card, \MSFC\ parallel interface card, and two \MOTS\ Spacewire cards.
A slot for an additional Spacewire card is included to accommodate two more cameras for the next \ESIS\ flight.
The C873 has a \SI{2.4}{\giga\hertz} Intel i7 processor with \SI{16}{\giga b} of memory.
The operating temperature range for the data system is -40 to +85 C.
The operating system for the flight data system is Linux Fedora 23.

The \DACS\ is responsible for several functions;
it controls the \ESIS\ experiment, responds to timers and uplinks, acquires and stores image data from the cameras, 
downlinks a subset of images through telemetry, and provides experiment health and status.
The \DACS\ is housed with the rest of the avionics (power supply, analog signal conditioning system) in a 
0.56-\SI{0.43}{\meter} transition section outside of the experiment section.
This relaxes the thermal and cleanliness constraints placed on the avionics.
Custom DC/DC converters are used for secondary voltages required by other electronic components.
The use of custom designed converters allowed additional ripple filtering for low noise."""
            ))

        with doc.create(pylatex.Subsection('Pointing System')):
            doc.append(pylatex.NoEscape(
                r"""
The imaging target will be selected prior to launch, the morning of the day of flight.
During flight, pointing will be maintained by the \SPARCS\ \citep{Lockheed69}.
Images from Camera 1 will be downlinked and displayed in real time on the \SPARCS\ control system console at intervals of 
$\sim$\SI{16}{\second} to verify pointing is maintained during flight."""
            ))

        with doc.create(pylatex.Subsection('Mechanical')):
            doc.append(pylatex.NoEscape(
                r"""
\ESIS\ and \MOSES\ are mounted on opposite sides of a composite optical table structure originally developed for the 
\SPDE~\citep{Bruner95lock}.
The layered carbon fiber structure features a convenient, precisely coplanar array of threaded inserts with precision 
counterbores.
The carbon fiber layup is designed to minimize the longitudinal coefficient of thermal expansion.
The optical table is housed in two \SI{0.56}{\meter} diameter skin sections, with a total length of \SI{3}{\meter}.
A ball joint and spindle assembly on one end and flexible metal aperture plate on the other hold the optical table in 
position inside the skin sections. 
The kinematic mounting system isolates the optical table from bending or twisting strain of the skins."""
            ))

    with doc.create(pylatex.Section('Mission Profile')):

        doc.append(pylatex.NoEscape(
            r"""
\ESIS\ will be launched aboard a sub-orbital Terrier Black Brant sounding rocket from White Sands Missile Range.
The experiment is currently scheduled for launch in August, 2019.
Trajectory will follow a roughly parabolic path, with $>$\SI{270}{\second} solar observing time above 
\SI{160}{\kilo\meter}.
\ESIS\ will begin continuously taking exposures at a fixed cadence immediately after launch, terminating just before the 
payload impacts the upper atmosphere.
Exposure length will be determined by the target selected for launch day.
Exposures taken while the payload shutter door is closed ($<$ \SI{160}{\kilo\meter}) will be used for dark calibration.
Data will be stored on board and downloaded after recovery, however a limited amount of data will be transmitted to the 
ground station via high speed telemetry as a safeguard against payload loss or destruction.
A parachute will slow the descent of the payload after it enters the atmosphere, and recovery will be accomplished by 
helicopter after the payload is located on the ground."""
        ))

        with doc.create(pylatex.Subsection(pylatex.NoEscape('\ESIS\ Mission Update'))):
            doc.append(pylatex.NoEscape(
                r"""
Since the time of writing \ESIS\ launched and was recovered successfully from White Sands Missile Range on 
September 30, 2019.
Unfortunately, due to failure of the mechanical shutter, no \MOSES\ data was obtained during this flight.
A  paper is forthcoming that will document the \ESIS\ instrument in its as-flown configuration~\citep{Courrier_inprep}.
A companion paper will describe \ESIS\ first results~\citep{Parker_inprep}.
Two significant changes, one to the \ESIS\ instrument and one to our alignment procedures, were made prior to launch and 
are summarized below.

The transfer from visible to \EUV\ grating alignment was completed by an alternative means.
The apparatus described by~\citet{Johnson18} was not able to maintain sufficient repeatability during test runs on 
diffraction grating surfaces.
To maintain the launch schedule, a phase shifting interferometer was used to transfer the alignment of the visible 
gratings to the \EUV\ flight gratings. 

A trade study was conducted, and it was decided to remove the primary aperture stop. The advantage was an increase in 
sensitivity.
The disadvantage was to sacrifice the unvignetted design described in Section \ref{subsec:AperturesandBaffles}.
The effective aperture is increased by a factor of 1.7 to 2.7 as a function of \FOV\ in the radial dimension.
The corresponding signal gradient is oriented along the dispersion direction of each channel;
vignetting increases (and signal decreases) when moving towards blue wavelengths 
(\ie\,moving to the left in Figure~\ref{fig:projections}).
This gradient is due almost entirely to vignetting by the central obscuration, and is linear across the entire \FOV.
The principal challenge is that the images cannot be corrected directly;
rather, since the gradient is repeated for each of the overlapping spectral line images, the vignetting can only be 
accounted for by forward modeling.
Since forward modeling is required for all of the inversion procedures under consideration for \ESIS\ data analysis, the 
vignetting was deemed low risk to the mission science."""
            ))

    with doc.create(pylatex.Section('Conclusions and Outlook')):
        doc.append(pylatex.NoEscape(
            r"""
\ESIS\ is a next generation slitless spectrograph, designed to obtain co-temporal spectral and spatial images of the solar 
transition region and corona.
In this report, we present details of the scientific objectives, instrument, image and spectral resolution, data 
acquisition, and flight profile.

\ESIS\ follows on the proven \MOSES\ design, incorporating several design changes to improve the utility of the instrument.
The symmetrical arrangement of \CCDs\ and diffraction gratings results in a compact instrument while increasing the number 
of dispersed images and dispersion planes.
This aids the inversion process, while also allowing access to higher order spectral line profile moments.
Individual gratings improve resolution by controlling aberration in each channel.
The addition of a field stop eliminates spectral contamination and provides an easily recognizable edge for data 
inversion.
The \ESIS\ design also demonstrates that all this can be accomplished in a volume small enough to serve as a prototype for 
a future orbital instrument.

For the first flight, four of the six available \ESIS\ channels will be populated with optics optimized around the 
O\,\textsc{v} emission line.
The large (\SI{11.3}{\arcminute} \roy{\fov}), high resolution \FOV\ (\SI{1.52}{\arcsecond} \roy{\spatialResolutionMax}, \SI{74}{\milli\angstrom} \roy{\spectralResolution}) can 
simultaneously observe the evolution of small scale \EUV\ flows and large scale \MHD\ waves in high temporal cadence. 
\ESIS\ also enables the study of transport of mass and energy in the transition region and corona during the $\sim 5$ 
minute data collection portion of rocket flight.

\ESIS\ was recovered after a successful first launch on September 30, 2019, with analysis of collected data currently 
in-process.
Subsequent flights will be proposed and the instrument refined with an eye toward orbital opportunities.
Suborbital flights will allow us to expand the instrument to its full complement of six channels and refine our data 
analysis methods, but do not provide access to major flares and eruptive events that drive space weather.
The long term prospect is that an \ESIS-like instrument on an orbital platform could provide high cadence maps of 
spectral line profiles in solar flares, allowing unique and comprehensive observations of the dynamics in solar eruptive 
events, flare ribbons, and the flare reconnection region."""
        ))

    doc.append(pylatex.Command('bibliography', arguments='sources'))

    return doc
