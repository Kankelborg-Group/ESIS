import aastex

__all__ = [
    'figure'
]


def figure() -> aastex.Figure:
    result = aastex.Figure("F-Baff1")
    result._star_latex_name = True
    result.add_image('figures/static/ESIS_Baffles.png', width=aastex.text_width_inches)
    result.add_caption(aastex.NoEscape(
        r"""Model view of \ESIS\ baffle placement and cutouts."""
    ))
    return result
