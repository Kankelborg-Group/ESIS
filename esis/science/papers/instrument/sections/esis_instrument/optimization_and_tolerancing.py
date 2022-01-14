import pylatex
import kgpy.latex
from ... import figures
from ... import tables

__all__ = [
    'subsection',
]


def subsection(doc: kgpy.latex.Document) -> pylatex.Subsection:
    result = pylatex.Subsection('Optimization and Tolerancing')
    result.escape = False
    result.append(
        r"""
The science resolution requirement of \angularResolutionRequirement (Table~\ref{table:scireq}) was 
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
    )
    # result.append(tables.surface_error.table_old())
    result.append(tables.surface_error.table())
    result.append(
        r"""
The initial grating radius of curvature, $R_g$, and ruling pattern of the \ESIS\ gratings were 
derived from the analytical equations developed by \citet{Poletto04} for stigmatic spectrometers.
A second order polynomial describes the ruling pattern,
\begin{equation} \label{Eq-d}
d = d_0 + d_1 r + d_2 r^2 \, ,
\end{equation}
where $r$ runs radially outward from the optical axis with its origin at the center of the grating \roy{shouldn't we be talking about $x$ here?}
(Fig.~\ref{fig:schematic}c).
The parameters of Equation~\ref{Eq-d} and $R_g$ were chosen so that the spatial and spectral focal curves intersect at 
the center of the O\,\textsc{v} \roy{\OV} image on the \CCD.

Starting from the analytically-derived optical prescription, a model of the system was developed in ray-trace \roy{raytrace} software.
Since the instrument is radially symmetric, only one grating and its associated lightpath was analyzed. \roy{delete previous sentence, all lightpaths were analyzed}
In the ray trace model, $R_g$, $d_1$, $d_2$, grating cant angle, \CCD\ cant angle, and focus position were then 
optimized to minimize the RMS spot at select positions in the O\,\textsc{v} \roy{\OV} \FOV, illustrated in Fig.~\ref{fig:psf}.
The optical prescription derived from the ray trace is listed in Table~\ref{table:prescription} and 
Figure~\ref{fig:schematic}. """
    )
    result.append(figures.psf.figure())
    result.append(figures.spot_size.figure())
    result.append(figures.focus_curve.figure())
    result.append(
        r"""
The ray trace model was also used to quantify how mirror and positional tolerances affect the 
instrument's spatial resolution.
Each element of the model was individually perturbed, then a compensation applied to adjust the image on the \CCD.
The compensation optimized grating tip/tilt angle and \CCD\ focus position, so that the image was re-centered and RMS 
spot size minimized at the positions in Fig.~\ref{F-spot} \roy{minimized at the vertices of the field stop and the central field angle}.
We then computed the maximum change in RMS spot size over all spot positions between the optimized and perturbed models.
The computed positional tolerances for each element in the \ESIS\ optical system are listed in Table~\ref{table:errorBudget}.

The imaging error budget is displayed in Table~\ref{table:errorBudget}.
For the primary mirror and grating surface figure contributions, we choose the \MTF\ figures of merit from the surface 
roughness specifications described earlier.
To quantify the remaining entries, we assume that each term can be represented by a gaussian function of width 
$\sigma^2$ that ``blurs'' the final image.
The value of $\sigma$ then corresponds to the maximum change in RMS spot size for each term as it is perturbed in the 
tolerance analysis described above.
The value of the \MTF\ in the right-most column of Table~\ref{table:errorBudget} is computed from 
each of the gaussian blur terms at the Nyquist frequency (\SI{0.5}{cycles\per arcsecond}).
From Table~\ref{table:errorBudget}, we estimate the total \MTF\ of \ESIS\ to be $0.109$ at the Nyquist frequency.
Compared to, for example, the Rayleigh criterion of \SI{0.09}{cycles\per arcsecond}~\citep{Rayleigh_1879} we estimate 
the resolution of \ESIS\ to be essentially pixel limited.
Since \ESIS\ pixels span \SI{0.76}{\arcsecond} \roy{\plateScaleMean}, the resolution target in Table~\ref{table:scireq} is obtained by this 
design."""
    )
    # result.append(pylatex.NoEscape(tables.error_budget.table_old))
    result.append(tables.error_budget.table(doc))
    return result
