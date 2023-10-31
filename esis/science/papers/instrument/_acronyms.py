import aastex
from aastex import NoEscape

__all__ = [
    "acronyms",
]


def acronyms() -> list[aastex.Acronym]:
    return [
        aastex.Acronym("ESIS", r"EUV Snapshot Imaging Spectrograph"),
        aastex.Acronym("MOSES", r"Multi-Order Solar EUV Spectrograph"),
        aastex.Acronym("TRACE", r"Transition Region and Coronal Explorer"),
        aastex.Acronym("AIA", r"Atmospheric Imaging Assembly"),
        aastex.Acronym("IRIS", r"Interface Region Imaging Spectrograph"),
        aastex.Acronym("MUSE", r"Multi-slit Solar Explorer"),
        aastex.Acronym("EIS", "EUV Imaging Spectrograph"),
        aastex.Acronym("EUNIS", "EUV Normal-incidence Spectrometer"),
        aastex.Acronym("SOHO", "Solar and Heliospheric Observatory"),
        aastex.Acronym("CDS", "Coronal Diagnostic Spectrometer", short=True),
        aastex.Acronym("MDI", r"Michelson Doppler Imager"),
        aastex.Acronym("CLASP", r"Chromospheric Lyman-alpha Spectro-polarimeter"),
        aastex.Acronym("HiC", r"High-Resolution Coronal Imager", "Hi-C"),
        aastex.Acronym("SXI", r"Solar X-ray Imager"),
        aastex.Acronym("GOES", r"Geostationary Operational Environmental Satellite"),
        aastex.Acronym("SPDE", r"Solar Plasma Diagnostics Experiment"),
        aastex.Acronym("FUV", "far ultraviolet"),
        aastex.Acronym("EUV", "extreme ultraviolet"),
        aastex.Acronym("TR", "transition region"),
        aastex.Acronym("CTIS", "computed tomography imaging spectrograph", plural=True),
        aastex.Acronym("FOV", "field of view"),
        aastex.Acronym("SNR", "signal-to-noise ratio", short=True),
        aastex.Acronym("PSF", "point-spread function", plural=True),
        aastex.Acronym("NRL", "Naval Research Laboratory"),
        aastex.Acronym("MHD", "magnetohydrodynamic"),
        aastex.Acronym("EE", "explosive event", plural=True),
        aastex.Acronym("QS", "quiet sun", short=True),
        aastex.Acronym("AR", "active region", short=True, plural=True),
        aastex.Acronym("CH", "coronal hole", short=True, plural=True),
        aastex.Acronym("CCD", "charge-coupled device", plural=True),
        aastex.Acronym("QE", "quantum efficiency", plural=True),
        aastex.Acronym("ULE", "ultra-low expansion"),
        aastex.Acronym("EDM", "electrical discharge machining"),
        aastex.Acronym("DEM", "differential emission measure"),
        aastex.Acronym("MTF", "modulation transfer function"),
        aastex.Acronym("LBNL", "Lawrence Berkley National Laboratory"),
        aastex.Acronym("MSFC", "Marshall Space Flight Center", short=True),
        aastex.Acronym("VR", NoEscape(r"\citet{Vernazza78}"), NoEscape(r"V\&R")),
        aastex.Acronym("DACS", "data acquisition and control system"),
        aastex.Acronym("GSE", "ground support equipment"),
        aastex.Acronym("MOTS", "military off-the-shelf"),
        aastex.Acronym(
            "SPARCS", "Solar Pointing Attitude Rocket Control System", short=True
        ),
        aastex.Acronym("SiC", "silicon carbide", short=True),
        aastex.Acronym("Al", "aluminum", short=True),
        aastex.Acronym("Mg", "magnesium", short=True),
        aastex.Acronym("Si", "silicon", short=True),
        aastex.Acronym("Cr", "chromium", short=True),
        aastex.Acronym("Ni", "nickel", short=True),
        aastex.Acronym("LN", "liquid nitrogen", NoEscape("LN$_2$")),
        aastex.Acronym("HeNe", "helium-neon", "He-Ne"),
    ]
