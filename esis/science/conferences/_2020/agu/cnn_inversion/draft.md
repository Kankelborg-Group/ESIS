# Convolutional Neural Networks for Tomographic Imaging Spectroscopy of the Solar Atmosphere

_Roy T. Smart, Charles C. Kankelborg, Jacob D. Parker and Nelson Goldsworth_

## Abstract

The _EUV Snapshot Imaging Spectrograph_ (ESIS) is an tomographic imaging spectrograph designed to measure spectral line 
profiles over a 2D field-of-view with much faster cadence than a rastering slit spectrograph.
ESIS uses four independent slitless spectrographs, each with a different dispersion direction but all fed from the same 
primary mirror.
To recover spectral line profiles from this arrangement, the images from each slitless spectrograph are interpreted
using computed tomography algorithms.
With only four independent spectrographs, this is a classic limited-angle tomography problem.
We trained a convolutional neural network to solve this tomography problem using observations from the 
_Interface Region Imaging Spectrograph_ (IRIS) as a training dataset.
We will present the performance of this network along with its application to the observations gathered during the
2019 ESIS sounding rocket flight.

## Introduction to ESIS

The _EUV Snapshot Imaging Spectrograph_ (ESIS) is a sounding rocket-based solar instrument which launched from White
Sands Missile Range in 2019.
ESIS has four independent slitless spectrographs (known as channels) with their dispersion directions oriented in 
multiples of 45 degrees.

In the figure above we can see the optical layout of one ESIS channel.
Light from the sun enters from the left and is focused by the parabolic primary on the right onto the octagonal field stop.
After the field stop, the image is dispersed and refocused onto the detector by a spherical diffraction grating.

The figure above is an image taken by ESIS during the 2019 flight.
O V 630 A is the octagon on the right, He I 584 A is the partial octagon on the left, and Mg X 610 A is the faint 
octagon in the middle.
This work will focus on analysis of the O V 630 A images.

## Distortion, Vignetting and Alignment

ESIS images have distortion, vignetting, and misalignement effects which need to be removed to compare images from different channels.
To achieve this, we developed a raytrace model of the instrument with 8 free parameters: detector piston/tip/tilt/roll and grating tip/tilt/roll/ruling density.
These free parameters were found by fitting the raytrace model to images from the 2019 flight using Powell's method.

The image above shows a difference between Channels 2 and 1 after correcting for distortion, vignetting and misalignment.
The raytrace model does not fully predict the vignetting function, leaving a gradient visible above.

## Inversion using CNNs

Recovering spectral line profiles from ESIS's four channels can be modeled as a limited-angle tomography problem.
In this work, we investigate solving this problem using a convolutional neural network (CNN)-based algorithm.
Our algorithm learns how to approximately solve this limited-angle tomography problem by training on sample problems
with known solutions.

In this implementation we used 56 full-disk mosaics of the Si IV 1394 A line gathered by the _Interface Region Imaging Spectrograph_ from 2013 - 2018 as a model of the O V 630 A line observed by ESIS.
These full-disk mosaics were used to create synthetic ESIS observations which were input into the CNN during training
The CNN was trained to minimize the RMS error between its output and the original full-disk mosaic.

## Training

## Validation

## Results

## Conclusion

