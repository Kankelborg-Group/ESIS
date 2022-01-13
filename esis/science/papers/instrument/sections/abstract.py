import kgpy.latex

__all__ = [
    'section'
]


def section() -> kgpy.latex.Abstract:
    result = kgpy.latex.Abstract()
    result.escape = False
    result.append(
        r"""
The \ESIS\ is a next generation rocket borne instrument designed to investigate magnetic reconnection 
and energy transport in the solar atmosphere by observing emission lines formed in the chromosphere (\HeI), 
the transition region (\OV), and corona (\MgX). 
The instrument is a pseudo Gregorian telescope with an octagonal field stop at prime focus.  
This field stop is re-imaged using an array of \numChannelsWords\ spherical diffraction gratings with differing 
dispersion angles oriented in $45^{\circ}$ increments, with each diffraction grating projecting the spectrum onto a 
unique detector.
The slitless multi-projection design will obtain co-temporal spatial (\plateScale) and spectral (\dispersion) images 
at high cadence ($>=$\detectorMinExposureLength). 
\amy{The instrument is designed to be capable of obtaining co-temporal spatial (\plateScale) and spectral 
(\dispersion) images at high cadence ($>=$\detectorMinExposureLength).}
Combining the co-temporal exposures from all the detectors will enable us to reconstruct line profile information 
at high spatial and spectral resolution over a large (\fov) \FOV. 
The instrument was launched on September 30, 2019.  The flight data is described in a subsequent paper. 
\acresetall"""
    )
    return result

