"""
Time Utilities
===============

Temporal calculations and timestamp handling for tracking data.

Key Functions:
- parse_timestamp: Parse various timestamp formats
- calculate_time_delta: Time difference with gap detection
- detect_time_gaps: Find significant gaps in a sequence
- is_monotonic: Check timestamp ordering
"""

from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple, Union
import re

from dateutil import parser as dateutil_parser
import numpy as np

from app.core.constants import TimeConstants


def parse_timestamp(
    value: Union[str, datetime, float, int],
    assume_utc: bool = True
) -> datetime:
    """
    Parse a timestamp from various formats into a datetime object.
    
    Handles:
    - ISO 8601 strings
    - Unix timestamps (int/float)
    - Common date formats
    - datetime objects (passthrough)
    
    Args:
        value: Timestamp in various formats
        assume_utc: If no timezone info, assume UTC
        
    Returns:
        datetime object (timezone-aware if possible)
        
    Raises:
        ValueError: If timestamp cannot be parsed
    """
    if isinstance(value, datetime):
        if value.tzinfo is None and assume_utc:
            return value.replace(tzinfo=timezone.utc)
        return value
    
    if isinstance(value, (int, float)):
        # Unix timestamp
        try:
            dt = datetime.fromtimestamp(value, tz=timezone.utc)
            return dt
        except (ValueError, OSError):
            # Try milliseconds
            return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
    
    if isinstance(value, str):
        # Try common formats first (faster than dateutil)
        for fmt in [
            TimeConstants.TIMESTAMP_FORMAT,
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y/%m/%d %H:%M:%S",
        ]:
            try:
                dt = datetime.strptime(value, fmt)
                if dt.tzinfo is None and assume_utc:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        
        # Fall back to dateutil (handles more formats)
        try:
            dt = dateutil_parser.parse(value)
            if dt.tzinfo is None and assume_utc:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            pass
    
    raise ValueError(f"Cannot parse timestamp: {value}")


def to_unix_timestamp(dt: datetime) -> float:
    """
    Convert datetime to Unix timestamp (seconds since epoch).
    
    Args:
        dt: datetime object
        
    Returns:
        Unix timestamp as float
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def calculate_time_delta(
    t1: datetime,
    t2: datetime
) -> float:
    """
    Calculate time difference in seconds.
    
    Args:
        t1: First timestamp
        t2: Second timestamp
        
    Returns:
        Difference in seconds (t2 - t1, can be negative)
    """
    return (t2 - t1).total_seconds()


def is_valid_time_delta(
    delta_seconds: float,
    min_seconds: float = TimeConstants.MIN_UPDATE_INTERVAL,
    max_seconds: float = TimeConstants.MAX_ACCEPTABLE_GAP
) -> bool:
    """
    Check if a time delta is within acceptable bounds.
    
    Args:
        delta_seconds: Time difference in seconds
        min_seconds: Minimum acceptable interval
        max_seconds: Maximum acceptable gap
        
    Returns:
        True if delta is valid
    """
    return min_seconds <= delta_seconds <= max_seconds


def detect_time_gaps(
    timestamps: List[datetime],
    gap_threshold_seconds: float = TimeConstants.MAX_NORMAL_GAP
) -> List[Tuple[int, int, float]]:
    """
    Detect significant time gaps in a sequence of timestamps.
    
    Args:
        timestamps: List of timestamps (should be sorted)
        gap_threshold_seconds: Minimum gap size to report
        
    Returns:
        List of tuples (start_index, end_index, gap_seconds)
    """
    gaps = []
    
    for i in range(1, len(timestamps)):
        delta = calculate_time_delta(timestamps[i-1], timestamps[i])
        if delta >= gap_threshold_seconds:
            gaps.append((i-1, i, delta))
    
    return gaps


def is_monotonic_increasing(
    timestamps: List[datetime],
    strict: bool = False
) -> bool:
    """
    Check if timestamps are monotonically increasing.
    
    Args:
        timestamps: List of timestamps
        strict: If True, requires strictly increasing (no duplicates)
        
    Returns:
        True if timestamps are monotonic
    """
    for i in range(1, len(timestamps)):
        delta = calculate_time_delta(timestamps[i-1], timestamps[i])
        if strict and delta <= 0:
            return False
        if not strict and delta < 0:
            return False
    return True


def find_duplicate_timestamps(
    timestamps: List[datetime],
    tolerance_seconds: float = 0.001
) -> List[List[int]]:
    """
    Find groups of duplicate or near-duplicate timestamps.
    
    Args:
        timestamps: List of timestamps
        tolerance_seconds: Maximum difference to consider duplicate
        
    Returns:
        List of groups (each group is list of indices)
    """
    if not timestamps:
        return []
    
    groups = []
    current_group = [0]
    
    for i in range(1, len(timestamps)):
        delta = abs(calculate_time_delta(timestamps[i-1], timestamps[i]))
        if delta <= tolerance_seconds:
            current_group.append(i)
        else:
            if len(current_group) > 1:
                groups.append(current_group)
            current_group = [i]
    
    if len(current_group) > 1:
        groups.append(current_group)
    
    return groups


def calculate_update_frequency(
    timestamps: List[datetime]
) -> Optional[float]:
    """
    Calculate average update frequency in Hz.
    
    Args:
        timestamps: List of timestamps
        
    Returns:
        Frequency in Hz, or None if insufficient data
    """
    if len(timestamps) < 2:
        return None
    
    total_time = calculate_time_delta(timestamps[0], timestamps[-1])
    if total_time <= 0:
        return None
    
    return (len(timestamps) - 1) / total_time


def segment_by_time_gap(
    timestamps: List[datetime],
    gap_threshold_seconds: float = TimeConstants.MAX_NORMAL_GAP
) -> List[Tuple[int, int]]:
    """
    Segment a timestamp sequence by time gaps.
    
    Useful for splitting a trip into continuous segments.
    
    Args:
        timestamps: List of timestamps
        gap_threshold_seconds: Gap size to split on
        
    Returns:
        List of (start_index, end_index) tuples for each segment
    """
    if not timestamps:
        return []
    
    segments = []
    segment_start = 0
    
    for i in range(1, len(timestamps)):
        delta = calculate_time_delta(timestamps[i-1], timestamps[i])
        if delta >= gap_threshold_seconds:
            segments.append((segment_start, i))
            segment_start = i
    
    segments.append((segment_start, len(timestamps)))
    return segments


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Human-readable string (e.g., "2h 15m 30s")
    """
    if seconds < 0:
        return f"-{format_duration(-seconds)}"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)


def time_window(
    center: datetime,
    window_minutes: int = 5
) -> Tuple[datetime, datetime]:
    """
    Create a time window around a center point.
    
    Args:
        center: Center timestamp
        window_minutes: Window size in minutes (total, not each side)
        
    Returns:
        Tuple of (start, end) datetimes
    """
    delta = timedelta(minutes=window_minutes / 2)
    return (center - delta, center + delta)


def vectorized_time_deltas(
    timestamps: np.ndarray
) -> np.ndarray:
    """
    Calculate time deltas between consecutive timestamps (vectorized).
    
    Args:
        timestamps: NumPy array of Unix timestamps
        
    Returns:
        Array of time deltas (first element is NaN)
    """
    deltas = np.empty(len(timestamps))
    deltas[0] = np.nan
    deltas[1:] = np.diff(timestamps)
    return deltas
