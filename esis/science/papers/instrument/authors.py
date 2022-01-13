import typing as typ
import kgpy.latex

__all__ = [
    'author_list'
]


def author_list() -> typ.List[kgpy.latex.aas.Author]:
    affil_msu = kgpy.latex.aas.Affiliation(
        'Montana State University, Department of Physics, '
        'P.O. Box 173840, Bozeman, MT 59717, USA'
    )
    affil_msfc = kgpy.latex.aas.Affiliation(
        'NASA Marshall Space Flight Center, '
        'Huntsville, AL 35812, USA'
    )
    affil_lbnl = kgpy.latex.aas.Affiliation(
        'Lawrence Berkeley National Laboratory, '
        '1 Cyclotron Road, Berkeley, CA 94720, USA'
    )
    affil_rxo = kgpy.latex.aas.Affiliation(
        'Reflective X-ray Optics LLC, '
        '425 Riverside Dr., #16G, New York, NY 10025, USA'
    )
    affil_gsfc = kgpy.latex.aas.Affiliation(
        'NASA Goddard Space Flight Center'
    )
    result = [
        kgpy.latex.aas.Author('Roy T. Smart', affil_msu),
        kgpy.latex.aas.Author('Hans T. Courrier', affil_msu),
        kgpy.latex.aas.Author('Jacob D. Parker', affil_gsfc),
        kgpy.latex.aas.Author('Charles C. Kankelborg', affil_msu),
        kgpy.latex.aas.Author('Amy R. Winebarger', affil_msfc),
        kgpy.latex.aas.Author('Ken Kobayashi', affil_msfc),
        kgpy.latex.aas.Author('Brent Beabout', affil_msfc),
        kgpy.latex.aas.Author('Dyana Beabout', affil_msfc),
        kgpy.latex.aas.Author('Ben Carrol', affil_msu),
        kgpy.latex.aas.Author('Jonathan Cirtain', affil_msfc),
        kgpy.latex.aas.Author('James A. Duffy', affil_msfc),
        kgpy.latex.aas.Author('Eric Gullikson', affil_lbnl),
        kgpy.latex.aas.Author('Micah Johnson', affil_msu),
        kgpy.latex.aas.Author('Laurel Rachmeler', affil_msfc),
        kgpy.latex.aas.Author('Larry Springer', affil_msu),
        kgpy.latex.aas.Author('David L. Windt', affil_rxo),
    ]
    return result
