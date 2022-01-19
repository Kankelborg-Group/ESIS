import pylatex
from ... import figures

__all__ = [
    'subsection'
]


def subsection() -> pylatex.Subsection:
    result = pylatex.Subsection('Coatings and Filters')
    result.escape = False
    result.append(figures.grating_multilayer_schematic.figure())
    result.append(figures.grating_efficiency_vs_angle.figure())
    result.append(figures.component_efficiency_vs_wavelength.figure())
    result.append(figures.grating_efficiency_vs_position.figure())
    result.append(
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
    )
    return result
