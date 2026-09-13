"""Idealized, pixel-replicated MSF-SPI reference implementation."""

from .patterns import AngleLibraryEntry, equivalent_illumination_pattern, load_angle_library
from .simulation import run_ideal_reconstruction

__all__ = [
    "AngleLibraryEntry",
    "equivalent_illumination_pattern",
    "load_angle_library",
    "run_ideal_reconstruction",
]
