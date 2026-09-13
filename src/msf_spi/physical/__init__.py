"""Pixel-resolved physical-SLM forward models and experiments."""

from .config import PhysicalSLMConfig, QualityThresholds
from .frequency_mixing import parallel_frequency_decomposition
from .order_model import simulate_recovered_pattern

__all__ = [
    "PhysicalSLMConfig",
    "QualityThresholds",
    "parallel_frequency_decomposition",
    "simulate_recovered_pattern",
]

