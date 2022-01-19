import pylatex
import kgpy.latex

__all__ = [
    'table'
]


def table() -> pylatex.Table:
    result = pylatex.Table(position='!htb')
    result._star_latex_name = True
    with result.create(pylatex.Center()) as centering:
        with centering.create(pylatex.Tabular(table_spec='llll', )) as tabular:
            tabular.escape = False
            tabular.add_row(['Parameter', 'Requirement', 'Science Driver', 'Capabilities'])
            tabular.add_hline()
            tabular.add_row([
                r'Spectral line',
                r'\OV',
                r'\EEs',
                r'\OVion, \MgXion, \HeIion, Figure~\ref{fig:bunch}',
            ])
            tabular.add_row([
                r'Spectral sampling',
                r'\spectralResolutionRequirement',
                r'Broadening from \MHD\ waves',
                r'\dispersionDoppler, Table~\ref{table:prescription}',
            ])
            tabular.add_row([
                r'Spatial resolution',
                r'\angularResolutionRequirement (\spatialResolutionRequirement)',
                r'\EEs',
                r'\spatialResolutionTotal, Table~\ref{table:errorBudget}',
            ])
            tabular.add_row([
                r'\SNRShort',
                r'\snrRequirement\ (\CHShort)',
                r'\MHD\ waves in \CHShort',
                r'\StackedCoronalHoleSNR\ ($\NumExpInStack \times \text{\detectorExposureLength}$ exp.), '
                r'Table~\ref{table:counts}',
            ])
            tabular.add_row([
                r'Cadence',
                r'\cadenceRequirement',
                r'Torsional waves',
                r'\detectorExposureLength\ eff., Section~\ref{subsec:SensitivityandCadence}',
            ])
            tabular.add_row([
                r'Observing time',
                r'\observingTimeRequirement',
                r'\EEs',
                r'\SI{270}{\second}, Section~\ref{sec:MissionProfile}',
            ])
            tabular.add_row([
                r'\FOV\ diameter',
                r'\fovRequirement',
                r'Span \QSShort, \ARShort, and limb',
                r'\fov, Table~\ref{table:prescription}',
            ])
    result.add_caption(pylatex.NoEscape(
        r"""\ESIS\ instrument requirements and capabilties. Note that MTF exceeds the Rayleigh criterion of 0.109."""
    ))
    result.append(kgpy.latex.Label('table:scireq'))
    return result
