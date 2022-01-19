import pylatex

__all__ = [
    'subsection',
]


def subsection() -> pylatex.Subsection:
    result = pylatex.Subsection('Magnetic Reconnection Events')
    result.escape = False
    result.append(
        r"""
Magnetic reconnection describes the re-arrangement of the magnetic topology wherein magnetic energy 
is converted to kinetic energy resulting in the acceleration of plasma particles.
Reconnection is implicated in many dynamic, high energy solar events.
Solar flares are a well studied example (\eg\,\citet{Priest02} and the references therein), however we have little hope 
of pointing in the right place at the right time to observe a significant flare event in a rocket flight lasting only 
five minutes.
Instead, we will search for signatures of magnetic reconnection in \TR\ spectral lines.      
%label to track table 1 references
\phantomsection
\label{t1_2}
A particular signature of reconnection in the \TR\ is the explosive energy release by ubiquitous, small scale events.
These \EEs\ are characterized as spatially compact ($\approx$\SI{1.5}{\mega\meter} length~\citep{Dere94}) line 
broadenings on the order of \SI{100}{\kilo\meter\per\second}~\citep{Dere91}.
They are observed across a range of \TR\ emission lines that span temperatures of \SI{20000}{}--\SI{250000}{\kelvin} 
(C\,\textsc{ii}--O\,\textsc{v})~\citep{1994Moses}.
The typical lifetime of an \EE\ is 60-\SI{90}{\second}~\citep{1994Moses,Dere94,Dere91}.
Due to their location near quiet sun magnetic network elements, and the presence of supersonic flows near the Alfv\`en 
speed, \citet{Dere91} first suggested that \EEs\ may result from the model of fast Petschek~\citep{Petschek64} 
reconnection. 

The spectral line profile of \EEs\ may indicate the type of reconnection that is occurring in the \TR\ 
(\eg\,\citet{Rust17}).
For example, the Petschek model of reconnection predicts a `bi-directional jet' line profile with highly Doppler 
shifted wings, but little emission from the line core~\citep{Innes99}.
\citet{Innes15} developed a reconnection model resulting from a plasmoid instability~\citep{Bhattacharjee09}.
In contrast to the bi-directional jet, this modeled line profile has bright core emission and broad wings.
Both types of profile are seen in slit spectrograph data (\eg, \citet{Innes97,Innes15}, and the references therein), 
however \MOSES\ observed \EEs\ with more complicated morphologies than either of these two models 
suggest~\citep{Fox10,Rust17}.
It is unclear whether the differing observations are a function of wavelength and temperature, a result of a limited 
number of observations, or because the morphology of the event is difficult to ascertain from slit spectrograph data.

%label to track table 1 references
\phantomsection
\label{t1_01}
\ESIS\ will observe magnetic reconnection in the context of \EEs, by extending the technique pioneered by \MOSES\ to 
additional \TR\ lines.
Explosive events are well suited to sounding rocket observations;
a significant portion of their temporal evolution can be captured in $>$\SI{150}{\second} (\eg\,the analysis by 
\citet{Rust17}) and they are sufficiently common to provide a statistically meaningful sample in a 5-minute rocket 
flight (\eg,~\citet{Dere89,Dere91}).
In similarity with \MOSES, we seek a \TR\ line for \ESIS\ that is bright and well enough isolated from neighboring 
emission lines so as to be easily distinguished."""
    )
    return result
