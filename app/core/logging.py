"""
Structured Logging Configuration
=================================

Production-grade logging setup using structlog.

Features:
- JSON logging for production (machine-parseable)
- Colored console output for development
- Request context tracking
- Performance metrics integration
- Exception handling with full context

Design Decisions:
- Use structlog for structured, context-rich logging
- JSON format in production for log aggregation (ELK, Datadog, etc.)
- Include request_id for distributed tracing
- Never log sensitive data (coordinates are OK, user PII is not)
"""

import logging
import sys
from typing import Any, Dict, Optional

import structlog
from structlog.types import Processor

from app.core.config import settings


def setup_logging() -> None:
    """
    Configure application logging.
    
    Sets up structlog with appropriate processors based on environment.
    Should be called once at application startup.
    """
    # Determine processors based on environment
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if settings.is_development and settings.log_format != "json":
        # Development: Pretty console output
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    else:
        # Production: JSON for log aggregation
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ]
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level),
    )
    
    # Set log levels for noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name, typically __name__ of the calling module
        
    Returns:
        Configured structlog logger instance
    """
    return structlog.get_logger(name)


class LogContext:
    """
    Context manager for adding temporary logging context.
    
    Usage:
        with LogContext(user_id=123, trip_id=456):
            logger.info("Processing trip")  # Includes user_id and trip_id
    """
    
    def __init__(self, **kwargs: Any):
        self.context = kwargs
        self._token = None
    
    def __enter__(self) -> "LogContext":
        self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self
    
    def __exit__(self, *args: Any) -> None:
        if self._token:
            structlog.contextvars.unbind_contextvars(*self.context.keys())


def log_processing_metrics(
    logger: structlog.stdlib.BoundLogger,
    operation: str,
    records_processed: int,
    duration_ms: float,
    errors: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Log standardized processing metrics.
    
    Use this for consistent metric logging across all data processing operations.
    
    Args:
        logger: Logger instance
        operation: Name of the operation (e.g., "ingest", "clean", "score")
        records_processed: Number of records processed
        duration_ms: Processing duration in milliseconds
        errors: Number of errors encountered
        extra: Additional context to include
    """
    metrics = {
        "operation": operation,
        "records_processed": records_processed,
        "duration_ms": round(duration_ms, 2),
        "records_per_second": round(records_processed / (duration_ms / 1000), 2) if duration_ms > 0 else 0,
        "errors": errors,
        "error_rate": round(errors / records_processed, 4) if records_processed > 0 else 0,
    }
    
    if extra:
        metrics.update(extra)
    
    logger.info("processing_complete", **metrics)


# Initialize logging on module import if not already done
_logging_initialized = False

def ensure_logging_initialized() -> None:
    """Ensure logging is initialized (idempotent)."""
    global _logging_initialized
    if not _logging_initialized:
        setup_logging()
        _logging_initialized = True
