# Convolutional Neural Networks for Tomographic Imaging Spectroscopy of the Solar Atmosphere


_Roy T. Smart, Charles C. Kankelborg, Jacob D. Parker and Nelson Goldsworth_

The _EUV Snapshot Imaging Spectrograph_ (ESIS) is an tomographic imaging spectrograph designed to measure spectral line 
profiles over a 2D field-of-view with much faster cadence than a rastering slit spectrograph.
ESIS uses four independent slitless spectrographs, each with a different dispersion direction but all fed from the same 
primary mirror.
To recover spectral line profiles from this arrangement, the images from each slitless spectrograph are interpreted
using computed tomography algorithms.
With only four independent spectrographs, this is a classic limited-angle tomography problem.
We trained a convolutional neural network to solve this tomography problem using observations from the 
_Coronal Diagnostic Spectrometer_ (CDS) as a training dataset.
We will present the performance of this network along with its application to the observations gathered during the
2019 ESIS sounding rocket flight.
