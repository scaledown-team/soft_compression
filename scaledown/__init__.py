"""
ScaleDown: Online Soft Compression And Reranking
Based on OSCAR paper (arXiv:2504.07109v1) with ModernBERT compressor
"""

__version__ = "0.1.0"

from .models import ScaleDownCompressor, ScaleDownGenerator, ScaleDownModel
from .config import ScaleDownConfig

__all__ = [
    "ScaleDownCompressor",
    "ScaleDownGenerator",
    "ScaleDownModel",
    "ScaleDownConfig",
]
