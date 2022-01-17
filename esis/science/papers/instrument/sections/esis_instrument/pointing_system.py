import pylatex

__all__ = [
    'subsection',
]


def subsection() -> pylatex.Subsection:
    result = pylatex.Subsection('Pointing System')
    result.escape = False
    result.append(
        r"""
The imaging target will be selected prior to launch, the morning of the day of flight.
During flight, pointing will be maintained by the \SPARCS\ \citep{Lockheed69}.
Images from Camera 1 will be downlinked and displayed in real time on the \SPARCS\ control system console at intervals of 
$\sim$\SI{16}{\second} to verify pointing is maintained during flight."""
    )
    return result
