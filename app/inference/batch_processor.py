"""
Batch Inference Processor
==========================

Optimized batch processing for Phase 2 inference.

Features:
- Efficient batching for GPU utilization
- Memory-aware batch sizing
- Async batch accumulation
- Performance monitoring
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from collections import deque
import time
import threading

import numpy as np

from app.core.logging import get_logger
from app.inference.accelerator import InferenceAccelerator, InferenceResult

logger = get_logger(__name__)


@dataclass
class InferenceBatch:
    """A batch of items for inference."""
    batch_id: str
    inputs: np.ndarray
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def size(self) -> int:
        return len(self.inputs)


@dataclass
class BatchResult:
    """Result from batch inference."""
    batch_id: str
    outputs: np.ndarray
    metadata: List[Dict[str, Any]]
    inference_time_ms: float
    queue_time_ms: float
    total_time_ms: float


class BatchInferenceProcessor:
    """
    Processes inference requests in optimized batches.
    
    Features:
    - Accumulates requests into batches
    - Triggers inference when batch full or timeout
    - Supports multiple models
    """
    
    def __init__(
        self,
        accelerator: InferenceAccelerator,
        default_batch_size: int = 100,
        max_wait_ms: float = 50.0,
    ):
        """
        Initialize batch processor.
        
        Args:
            accelerator: Inference accelerator
            default_batch_size: Default batch size
            max_wait_ms: Maximum wait before processing partial batch
        """
        self.accelerator = accelerator
        self.batch_size = default_batch_size
        self.max_wait_ms = max_wait_ms
        
        self._pending: Dict[str, deque] = {}  # model -> pending items
        self._batch_counter = 0
        self._stats = {
            "batches_processed": 0,
            "items_processed": 0,
            "total_inference_time_ms": 0.0,
            "avg_batch_size": 0.0,
        }
    
    def process_single(
        self,
        model_name: str,
        inputs: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InferenceResult:
        """
        Process a single inference request.
        
        Args:
            model_name: Model to use
            inputs: Input data (single sample or small batch)
            metadata: Optional metadata
            
        Returns:
            InferenceResult
        """
        return self.accelerator.infer(model_name, inputs)
    
    def process_batch(
        self,
        model_name: str,
        batch: InferenceBatch,
    ) -> BatchResult:
        """
        Process a prepared batch.
        
        Args:
            model_name: Model to use
            batch: Prepared batch
            
        Returns:
            BatchResult
        """
        queue_time = (datetime.utcnow() - batch.created_at).total_seconds() * 1000
        
        start = time.time()
        result = self.accelerator.infer(model_name, batch.inputs)
        total_time = (time.time() - start) * 1000
        
        # Update stats
        self._stats["batches_processed"] += 1
        self._stats["items_processed"] += batch.size
        self._stats["total_inference_time_ms"] += result.inference_time_ms
        self._stats["avg_batch_size"] = (
            self._stats["items_processed"] / self._stats["batches_processed"]
        )
        
        return BatchResult(
            batch_id=batch.batch_id,
            outputs=result.outputs,
            metadata=batch.metadata,
            inference_time_ms=result.inference_time_ms,
            queue_time_ms=queue_time,
            total_time_ms=total_time,
        )
    
    def create_batches(
        self,
        inputs: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: Optional[int] = None,
    ) -> List[InferenceBatch]:
        """
        Split inputs into batches.
        
        Args:
            inputs: Input array
            metadata: Optional metadata for each sample
            batch_size: Override batch size
            
        Returns:
            List of InferenceBatch objects
        """
        bs = batch_size or self.batch_size
        batches = []
        
        for i in range(0, len(inputs), bs):
            batch_inputs = inputs[i:i + bs]
            batch_metadata = metadata[i:i + bs] if metadata else []
            
            self._batch_counter += 1
            
            batches.append(InferenceBatch(
                batch_id=f"batch_{self._batch_counter}",
                inputs=batch_inputs,
                metadata=batch_metadata,
            ))
        
        return batches
    
    def process_all(
        self,
        model_name: str,
        inputs: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> List[BatchResult]:
        """
        Process all inputs in optimal batches.
        
        Args:
            model_name: Model to use
            inputs: All inputs
            metadata: Optional metadata
            
        Returns:
            List of BatchResults
        """
        batches = self.create_batches(inputs, metadata)
        
        results = []
        for batch in batches:
            result = self.process_batch(model_name, batch)
            results.append(result)
        
        return results
    
    def get_concatenated_outputs(
        self,
        results: List[BatchResult],
    ) -> np.ndarray:
        """Concatenate outputs from multiple batch results."""
        return np.concatenate([r.outputs for r in results])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = dict(self._stats)
        
        if stats["batches_processed"] > 0:
            stats["avg_inference_time_ms"] = (
                stats["total_inference_time_ms"] / stats["batches_processed"]
            )
        
        return stats


class AsyncBatchAccumulator:
    """
    Accumulates requests for batch processing.
    
    Useful for high-throughput scenarios where
    requests arrive continuously.
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        max_wait_ms: float = 50.0,
        on_batch_ready: Optional[Callable[[InferenceBatch], None]] = None,
    ):
        """
        Initialize accumulator.
        
        Args:
            batch_size: Target batch size
            max_wait_ms: Max wait before processing
            on_batch_ready: Callback when batch ready
        """
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.on_batch_ready = on_batch_ready
        
        self._buffer: List[Tuple[np.ndarray, Dict[str, Any]]] = []
        self._lock = threading.Lock()
        self._batch_counter = 0
    
    def add(
        self,
        inputs: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[InferenceBatch]:
        """
        Add item to accumulator.
        
        Args:
            inputs: Input data (single sample)
            metadata: Optional metadata
            
        Returns:
            InferenceBatch if batch is ready, None otherwise
        """
        with self._lock:
            self._buffer.append((inputs, metadata or {}))
            
            if len(self._buffer) >= self.batch_size:
                return self._flush()
        
        return None
    
    def _flush(self) -> InferenceBatch:
        """Flush buffer into a batch."""
        self._batch_counter += 1
        
        inputs = np.array([item[0] for item in self._buffer])
        metadata = [item[1] for item in self._buffer]
        
        batch = InferenceBatch(
            batch_id=f"async_batch_{self._batch_counter}",
            inputs=inputs,
            metadata=metadata,
        )
        
        self._buffer = []
        
        if self.on_batch_ready:
            self.on_batch_ready(batch)
        
        return batch
    
    def flush(self) -> Optional[InferenceBatch]:
        """Force flush current buffer."""
        with self._lock:
            if self._buffer:
                return self._flush()
        return None
    
    @property
    def pending_count(self) -> int:
        """Get number of pending items."""
        return len(self._buffer)
