import pylatex
from ... import figures

__all__ = [
    'subsection',
]


def subsection() -> pylatex.Subsection:
    result = pylatex.Subsection('Alignment and Focus')
    result.escape = False
    result.append(
        r"""
In the conceptual phase of \ESIS, the decision was made to perform focus and alignment in visible light with a \HeNe\ 
source.
Certain difficulties are introduced by this choice, however, the benefits outweigh the operational complexity and 
equipment that would be required for focus in \EUV.
Moreover, a sounding rocket instrument requires robust, adjustment-free mounts to survive the launch environment.
Such a design is not amenable to iterative adjustment in vacuum.  The choice of alignment wavelength is arbitrary for 
most components;
\CCD\ response and multilayer coating reflectively is sufficient across a wide band a visible wavelengths.
The exceptions are the thin film filters (which will not be installed until just before launch and have no effect on 
telescope alignment and focus) and the diffraction gratings.
Visible light gratings have been manufactured specifically for alignment and focus.
These gratings are identical to the \EUV\ flight version, but with a ruling pattern scaled to a 
\SI{632.8}{\nano\meter} wavelength."""
    )
    result.append(figures.alignment_transfer.figure())
    result.append(
        r"""
After alignment and focus has been obtained with the \HeNe\ source, the instrument will be prepared for flight by 
replacing the visible gratings by the \EUV\ flight versions.
Each grating (\EUV\ and alignment) is individually mounted to a backing plate using a bipod system similar to that of the 
primary mirror.
The array of gratings on their backing plates are in turn mounted to a single part which we call the `tuffet.'
The backing plate is attached to the tuffet by bolts through spherical washers.
With this mounting scheme, the gratings can be individually aligned in tip/tilt.
The tuffet is attached to the secondary mirror mount structure (Fig.~\ref{F-alt}).
This enables the entire grating array to be replaced simply by removing/installing the tuffet, such as when switching 
between alignment and \EUV\ gratings.

Properly positioning the gratings will be the most difficult aspect of final telescope assembly.
Table~\ref{table:tol} shows that the telescope is very sensitive to defocus.
The depth of field is an order of magnitude smaller in \EUV\ than in visible light.
Moreover, the sensitivity to defocus at the gratings is $M^2+1=17$ times greater than at the detectors.
Another sensitive aspect of the telescope is grating tip/tilt.
A tolerance of $\sim\pm$\SI{0.5}{\milli\radian} will be needed to ensure that the entire image lands on the active area 
of the \CCD.

%Once the visible gratings are aligned and focused, the challenge is to transfer this alignment to the UV gratings.
%\citet{Johnson18} describes a procedure and the apparatus constructed to accurately transfer the position of an 
%alignment grating radius to an \EUV\ flight grating.
%This device, displayed in Fig.~\ref{F-alt}, consists of an array of three miniature confocal microscopes that record the 
%position of the alignment grating radius.
%The alignment grating is replaced by an \EUV\ grating, which is then measured into position by the same apparatus.
%This device (and procedure) is capable of obtaining position measurements of the diffraction gratings to a repeatability 
%of $\approx$\SI{14}{\micro\meter} in the three confocal channels.
%This alignment transfer apparatus will ensure that the \EUV\ flight gratings are positioned to within the tolerances 
%described in Table~\ref{table:tol}.

Once the visible gratings were aligned and focused, the challenge is to transfer this alignment to the \EUV\ flight 
gratings.
We performed this transfer using a 4D Phasecam 6000 interferometer.
We aligned the interferometer to each visible grating such that the tip, tilt, and defocus aberrations were zero.
Then, placing the corresponding flight grating in the visible grating's place, we shimmed the mounting screws of the 
flight grating to match the tip, tilt, and defocus of the alignment grating.
Since the mounting interface for the tuffet is extremely repeatable by design we were able to swap tuffets and compare 
their alignment and focus to ensure everything was transferred correctly.

When transferring the focus and alignment two key details were considered.
First, there is nothing about the tuffet that constrains the grating roll.
Therefore, we needed to ensure each flight grating had the same roll as each visible grating.
This was accomplished by using a \HeNe\ laser diverged through a cylindrical optic that illuminated each grating with a 
line perpendicular to the grating blaze direction.
The line of light was reflected back onto a ruled target that could be compared between gratings.
Since our alignment gratings were ruled to image light of approximately an order of magnitude longer wavelength the
laser and cylindrical optic were placed at the position of Littrow for the 10th order image in the visible grating,
and the 1st order for the \EUV\ gratings.
Second, during testing we measured slight differences in radius of curvature between each grating. 
Therefore each flight grating was prescribed a specific amount of defocus to account for the difference in radius of
curvature between each optic when transferring alignment and focus."""
    )
    return result
