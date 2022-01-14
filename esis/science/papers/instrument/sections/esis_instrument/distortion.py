import pylatex
from ... import figures
from ... import tables

__all__ = [
    'subsection',
]


def subsection() -> pylatex.Subsection:
    result = pylatex.Subsection('Distortion')
    result.escape = False
    result.append(
        r"""
The distortion is due to two factors: first, the tilt of the detector as needed to maintain good focus over the \FOV 
\citep{Poletto04}; second, the anamorphic magnification of the grating (see \cite{Schweizer1979})."""
    )
    result.append(
        r"""
\begin{equation}
\begin{split}
\left(x', y'\right) &= \C + \C_x x + \C_y y + \C_\lambda \lambda \\
&+ \C_{xx} x^2 + \C_{xy} x y + \C_{y \lambda} x \lambda \\
&+ \C_{yy} y^2 + \C_{y \lambda} y \lambda + \C_{\lambda \lambda} \lambda^2
\end{split}
\end{equation}"""
    )
    result.append(tables.distortion.table())
    result.append(figures.distortion.figure())
    result.append(figures.distortion_residual.figure())
    return result
