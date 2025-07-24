"""
Models of the optical coatings and substrates.
"""

from ._materials import (
    multilayer_design,
    multilayer_witness_measured,
    multilayer_witness_fit,
    multilayer_fit,
)

__all__ = [
    "multilayer_design",
    "multilayer_witness_measured",
    "multilayer_witness_fit",
    "multilayer_fit",
]
