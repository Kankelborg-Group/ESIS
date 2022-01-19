import pylatex
import kgpy.latex
import esis.optics

__all__ = [
    'table_old',
]


def table_old() -> str:
    return r"""
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


def table() -> pylatex.Table:
    optics_all = esis.flight.optics.as_measured()
    detector = optics_all.detector
    result = pylatex.Table()
    with result.create(pylatex.Center()) as centering:
        with centering.create(pylatex.Tabular('ccccc')) as tabular:
            tabular.escape = False
            tabular.add_row([r'Channel', r'Quad.', r'Gain', r'Read noise', r'Dark current', ])
            tabular.add_row(
                ['', '', f'({detector.gain.unit:latex_inline})', f'({detector.readout_noise.unit:latex_inline})',
                 f'({detector.dark_current.unit:latex_inline})'])
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
    result.add_caption(pylatex.NoEscape(r'\ESIS\ camera properties'))
    result.append(kgpy.latex.Label('tabel:cameraProperties'))
    return result
