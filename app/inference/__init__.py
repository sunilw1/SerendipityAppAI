"""
Phase 2 Inference Module
=========================

GPU-accelerated inference infrastructure for Phase 2.

Supports:
- TensorRT optimization
- Batch inference
- Model versioning
- Triton integration (optional)

Design Principles:
- TensorRT used where useful
- GPU acceleration for batch inference
- Services kept modular
- Triton only introduced if scale requires
"""

from app.inference.accelerator import (
    InferenceAccelerator,
    GPUInferenceEngine,
    CPUFallbackEngine,
)
from app.inference.batch_processor import (
    BatchInferenceProcessor,
    InferenceBatch,
)

__all__ = [
    "InferenceAccelerator",
    "GPUInferenceEngine",
    "CPUFallbackEngine",
    "BatchInferenceProcessor",
    "InferenceBatch",
]
