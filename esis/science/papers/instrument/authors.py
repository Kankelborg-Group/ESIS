import typing as typ
import aastex

__all__ = [
    'author_list'
]


def author_list() -> typ.List[aastex.Author]:
    affil_msu = aastex.Affiliation(
        'Montana State University, Department of Physics, '
        'P.O. Box 173840, Bozeman, MT 59717, USA'
    )
    affil_msfc = aastex.Affiliation(
        'NASA Marshall Space Flight Center, '
        'Huntsville, AL 35812, USA'
    )
    affil_lbnl = aastex.Affiliation(
        'Lawrence Berkeley National Laboratory, '
        '1 Cyclotron Road, Berkeley, CA 94720, USA'
    )
    affil_rxo = aastex.Affiliation(
        'Reflective X-ray Optics LLC, '
        '425 Riverside Dr., #16G, New York, NY 10025, USA'
    )
    affil_gsfc = aastex.Affiliation(
        'NASA Goddard Space Flight Center'
    )
    result = [
        aastex.Author('Roy T. Smart', affil_msu),
        aastex.Author('Hans T. Courrier', affil_msu),
        aastex.Author('Jacob D. Parker', affil_gsfc),
        aastex.Author('Charles C. Kankelborg', affil_msu),
        aastex.Author('Amy R. Winebarger', affil_msfc),
        aastex.Author('Ken Kobayashi', affil_msfc),
        aastex.Author('Brent Beabout', affil_msfc),
        aastex.Author('Dyana Beabout', affil_msfc),
        aastex.Author('Ben Carrol', affil_msu),
        aastex.Author('Jonathan Cirtain', affil_msfc),
        aastex.Author('James A. Duffy', affil_msfc),
        aastex.Author('Eric Gullikson', affil_lbnl),
        aastex.Author('Micah Johnson', affil_msu),
        aastex.Author('Laurel Rachmeler', affil_msfc),
        aastex.Author('Larry Springer', affil_msu),
        aastex.Author('David L. Windt', affil_rxo),
    ]
    return result
