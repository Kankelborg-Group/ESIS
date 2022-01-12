import typing as typ
import pylatex
from . import acronyms

__all__ = ['body']


def body() -> typ.List[pylatex.base_classes.LatexObject]:

    result = []

    result.append(pylatex.NoEscape(
        '\\savesymbol{tablenum}\n'
        '\\usepackage{siunitx}\n'
        '\\restoresymbol{SIX}{tablenum}\n'
    ))

    result.append(pylatex.NoEscape(
        r"""
\makeatletter
\newcommand{\acposs}[1]{%
 \expandafter\ifx\csname AC@#1\endcsname\AC@used
   \acs{#1}'s%
 \else
   \aclu{#1}'s (\acs{#1}'s)%
 \fi
}
\newcommand{\Acposs}[1]{%
 \expandafter\ifx\csname AC@#1\endcsname\AC@used
   \acs{#1}'s%
 \else
   \Aclu{#1}'s (\acs{#1}'s)%
 \fi
}
\makeatother"""
    ))

    result.append(pylatex.NoEscape(r'\newcommand{\amy}[1]{{{\color{red} #1}}}'))
    result.append(pylatex.NoEscape(r'\newcommand{\jake}[1]{{{\color{purple} #1}}}'))
    result.append(pylatex.NoEscape(r'\newcommand{\roy}[1]{{{\color{blue} #1}}}'))

    result.append(pylatex.Command('bibliographystyle', 'aasjournal'))

    result += acronyms.all()

    result.append(pylatex.Command('DeclareSIUnit', [pylatex.NoEscape(r'\angstrom'), pylatex.NoEscape(r'\AA')]))

    return result
