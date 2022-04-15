import pathlib
import pylatex
import kgpy.latex


def save_document() -> None:

    doc = kgpy.latex.Document(
        default_filepath=str(pathlib.Path(__file__).parent / 'abstract'),
        documentclass='aastex631',
        document_options=['twocolumn'],
    )

    title = kgpy.latex.Title('Deep Learning Techniques for Inverting Observations from the EUV Snapshot Imaging Spectrograph')
    doc.append(title)

    affil_msu = kgpy.latex.aas.Affiliation(
        'Montana State University, Department of Physics, '
        'P.O. Box 173840, Bozeman, MT 59717, USA'
    )

    affil_gsfc = kgpy.latex.aas.Affiliation(
        'NASA Goddard Space Flight Center'
    )

    doc.append(kgpy.latex.aas.Author('Roy T. Smart', affilation=affil_msu))
    doc.append(kgpy.latex.aas.Author('Charles C. Kankelborg', affilation=affil_msu))
    doc.append(kgpy.latex.aas.Author('Jacob D. Parker', affilation=affil_gsfc))

    abstract = kgpy.latex.Abstract()
    abstract.escape = False
    abstract.append(
        r"""Imaging spectroscopy of the solar atmosphere with high spatial, spectral, and temporal resolution over a
wide field of view is a longstanding goal of heliophysics because it allows for the measurement of important plasma parameters
such as velocity and density with high fidelity.
The EUV Snapshot Imaging Spectrograph (ESIS) is a sounding rocket instrument designed to capture EUV spectral line
profiles over a large, 2D field of view with much higher temporal resolution than current rastering slit spectrographs.
ESIS achieves this using a computed tomography imaging spectrograph design, with four channels.
Each channel is an independent slitless spectrograph, illuminated by a common primary mirror, but oriented with a 
unique dispersion direction.
Each ESIS exposure, comprising of four channels, can be inverted to recover spectral line profiles for every point
in the field of view using limited-angle computed tomography techniques.
In this work we present progress on the development of a deep learning algorithm that learns to solve the ESIS
limited-angle tomography problem using data from the Interface Region Imaging Spectrograph as a training dataset.
We will apply this algorithm to observations from the 2019 ESIS flight and compare the results to those obtained
using previous methods.
"""
    )
    doc.append(abstract)

    dummy_section = pylatex.Section(' ')
    dummy_section.append(' ')
    doc.append(dummy_section)

    doc.generate_pdf()
