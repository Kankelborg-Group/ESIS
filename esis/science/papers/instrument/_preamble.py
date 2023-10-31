import aastex
from . import acronyms

__all__ = ["preamble"]


def preamble() -> list[str]:

    result = []

    result.append(aastex.NoEscape(
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

    result.append(aastex.NoEscape(r'\newcommand{\amy}[1]{{{\color{red} #1}}}'))
    result.append(aastex.NoEscape(r'\newcommand{\jake}[1]{{{\color{purple} #1}}}'))
    result.append(aastex.NoEscape(r'\newcommand{\roy}[1]{{{\color{blue} #1}}}'))

    result += acronyms()

    result.append(aastex.Command('DeclareSIUnit', [aastex.NoEscape(r'\angstrom'), aastex.NoEscape(r'\AA')]))

    return result
