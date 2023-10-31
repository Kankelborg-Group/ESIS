import aastex

__all__ = [
    'figure',
]


def figure() -> aastex.Figure:
    result = aastex.Figure("F-cameras")
    result.add_image('figures/old/ESIS_Cameras_1gr_text', width=aastex.column_width_inches)
    result.add_caption(aastex.NoEscape(
        r"""
\ESIS\ camera assembly as built by \MSFCShort.  
Thin film filters and filter tubes are not installed in this image."""
    ))
    return result
