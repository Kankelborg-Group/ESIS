import pylatex
from ... import figures
from ... import tables

__all__ = [
    'subsection',
]


def subsection() -> pylatex.Subsection:
    result = pylatex.Subsection('Cameras')
    result.escape = False
    result.append(figures.cameras.figure())
    result.append(
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
\citet{Stern04} for \GOES\ \SXI\ instruments."""
    )
    # result.append(tables.cameras.table_old())
    result.append(tables.cameras.table())
    return result
