import numpy as np
import astropy.units as u
import pylatex
import kgpy.latex
import kgpy.format
from .. import optics

__all__ = [
    'table_old',
    'table',
]

table_old = pylatex.NoEscape(
    r"""
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
\end{table}"""
)


def table(doc: kgpy.latex.Document) -> pylatex.Table:
    result = pylatex.Table()

    optics_single = optics.as_designed_single_channel()
    wavelength = optics_single.bunch.wavelength
    index_o5 = np.nonzero(optics_single.bunch.ion == 'o_5')[0][0]
    wavelength_o5 = wavelength[index_o5]
    index_mg10_2 = np.nonzero(optics_single.bunch.ion == 'mg_10')[0][1]
    wavelength_mg10_2 = wavelength[index_mg10_2]

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

    with result.create(pylatex.Center()) as centering:
        with centering.create(pylatex.Tabular('lrrrr')) as tabular:
            tabular.escape = False
            tabular.add_row([r'Source', r'\VR', r'\VR', r'\VR', r'\CDS'])
            tabular.add_row(r'Solar context', r'\QSShort', r'\CHShort', r'\ARShort', r'\ARShort')
            tabular.add_hline()
            tabular.add_hline()
            tabular.append(f'\\multicolumn{{5}}{{c}}{{{label}}}\\\\')
            tabular.add_row([r'\OV', ] + [f'{c:0.0f}' for c in counts_o5.value])
            tabular.add_row([r'\MgXdim', ] + [f'{c:0.0f}' for c in counts_mg10.value])
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

    result.add_caption(pylatex.NoEscape(
        r"""
Estimated signal statistics per channel (in photon counts) for \ESIS\ lines in \CH, \QS, and \AR.
Note that the \SNR\ estimates are lower bounds since charge diffusion decreases the shot noise."""
    ))
    result.append(kgpy.latex.Label('table:counts'))
    return result
