import pylatex

__all__ = [
    'section'
]


def section() -> pylatex.Section:
    result = pylatex.Section('Conclusions and Outlook')
    result.escape = False
    result.append(
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
    )
    return result
