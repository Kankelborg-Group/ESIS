import pylatex

__all__ = [
    'subsection',
]


def subsection() -> pylatex.Subsection:
    result = pylatex.Subsection('Energy Transfer')
    result.escape = False
    result.append(
        r"""
Tracking the mass and energy flow through the solar atmosphere is a long-standing goal in solar physics.
Bulk mass flow is evidenced by Doppler shifts or skewness in spectral lines.
However, the observed non-thermal broadening of \TR\ spectral lines may result from a variety of physical processes, 
including \MHD\ waves~\citep{DePontieu15, DePontieu07}, high-speed evaporative up-flows (\eg\,nanoflares, 
\citet{Patsourakos06}), turbulence, and other sources (\eg\,\citet{Mariska1992}).
This is a broad topic which \ESIS\ can address in many ways.
Here we will focus on a single application;
\ESIS\ will search for sources of Alfv\'en waves in the solar atmosphere by observing line broadening as the 
spectroscopic signature of these waves.

Alfv\'en waves in coronal holes are observed to carry an energy flux of 
\SI{7e5}{erg\per\centi\square\meter\per\second}, enough to energize the fast solar wind \citep{Hahn2012,Hahn2013}.
The source and frequency spectrum of these waves is unknown.
Here, we hypothesize that \MHD\ waves are similarly ubiquitous in quiet Sun and active regions, and play an important 
role in the energization of the quiescent corona.

%label to track table 1 references
\phantomsection
\label{t1_1}
The magnitude of non-thermal broadening of optically thin spectral lines is a direct measure of the wave 
amplitude~\citep{Banerjee09,Hahn2012,Hahn2013}.
We may estimate a lower limit on the non-thermal velocity to be observed as follows.
We assume that the magnetic field is constant for small changes in scale height in the \TR\ and that line of sight 
effects are negligible for observations sufficiently far from disk center.
Since the solar wind is not accelerated to an appreciable fraction of the Alfv\'en wave velocity at altitudes below 
$R \leq 1.15R_\odot$~\citep{Cranmer05}, the wave amplitude, $v_{nt}$, depends only weakly on electron density, $n_e$, so 
that $v_{nt} \propto n_e^{-1/4}$~\citep{Hahn2013,Moran01}.
Assuming pressure balance between the low corona and transition zone, we may infer non-thermal velocities in the \TR\ by 
scaling according to the temperature drop, $v_{nt} \propto T^{1/4}$.
The measured non-thermal velocity of \SI{24}{\kilo\meter\per\second} for Si\,\textsc{viii}~\citep{Doyle98} 
(\SI{0.8}{\mega\kelvin}~\citep{Moran03}) near the limb should, neglecting damping, correspond to velocities of at least 
\SI{21}{\kilo\meter\per\second} in mid \TR\ Ne\,\textsc{vii}, and \SI{18}{\kilo\meter\per\second} in the lower 
O\,\textsc{v} (\SI{0.25}{\mega\kelvin}) line.
The above non-thermal velocities are arrived at assuming both O\,\textsc{v} and Ne\,\textsc{vii} are formed near their 
ionization equilibrium temperatures.
For O\,\textsc{v}, the thermal width is $\sim$\SI{11}{\kilo\meter\per\s} at \SI{0.25}{\mega\kelvin} which means the 
total linewidth is primarily due to the non-thermal component.    
    
More recently, ~\citet{Srivastava17} observed torsional Alfv\'en waves with amplitude 
$\sim$\SI{20}{\kilo\meter\per\second} and period $\sim$\SI{30}{\second} in the chromosphere.
Modeling shows that these torsional waves can transfer a significant amount of energy to the corona~\citep{Kudoh99}.
The torsional motion will be observed as Doppler shifts when viewed from the side.
The oscillation period is long enough to be well resolved but short enough to see $\sim$\SI{10}{} cycles in a single 
rocket flight.
An \ESIS-like instrument is therefore well suited to observations of torsional Alfv\'en wave propagation over multiple 
heights in the \TR. 

By mapping Doppler velocities over a wide field of view in the \TR, \ESIS\ can address questions about both the origin 
of waves and whether they are able to propagate upward into the corona.
Independent of the two propagation modes discussed above, there is a range of possible sources for Alfv\'en 
(and other \MHD) waves in the solar atmosphere.
Three potential scenarios are: \begin{inparaenum}[(1)] \item Waves originate in the chromosphere or below and propagate 
through the \TR\ at a spatially uniform intensity; \label{wave-1}
\item Intense sources are localized in the \TR, but fill only a fraction of the surface\label{wave-2}; and \item Weak 
sources are localized in the \TR, but cover the surface densely enough to appear like the first case\label{wave-3}. 
\end{inparaenum}
The resulting non-thermal widths for localized sources will be significantly higher than the 
$\sim$\SI{20}{\kilo\meter\per\second} mean derived above.
The concentration of non-thermal energy observed by \ESIS\ will serve as an indicator of source density.
Comparison of Doppler maps captured at different temperatures by \ESIS\ will indicate whether a uniform source density originates in the 
chromosphere or below (scenario~\ref{wave-1}) or is associated with spatially distributed \TR\ phenomena 
(scenario~\ref{wave-3}) such as explosive events, or macrospicules.
Comparison with a wider selection of ground and space based imagery will allow us to determine whether intense, 
localized sources (scenario~\ref{wave-2}) are associated with converging or emerging magnetic bipoles, type \textsc{ii} 
spicules, spicule bushes, or other sources beneath the \TR.
For these comparisons, we need only to localize, rather than resolve, wave sources.
A spatial resolution of $\sim$\SI{2}{\mega\meter} will be sufficient to localize sources associated with magnetic flux 
tubes that are rooted in photospheric inter-granular network lanes (\eg\,\citet{Berger95ApJ})."""
    )
    return result
