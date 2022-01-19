import pylatex

__all__ = [
    'subsection',
]


def subsection() -> pylatex.Subsection:
    result = pylatex.Subsection('Avionics')
    result.escape = False
    result.append(
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
    )
    return result
