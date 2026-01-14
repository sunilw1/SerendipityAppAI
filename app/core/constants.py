"""
Application Constants
======================

Centralized constants for the Serendipity AI Backend.

Design Philosophy:
- All magic numbers are named and documented
- Physical constants are based on real-world limits
- Thresholds are conservative to avoid false positives
- Values are tuned based on GPS and mobility research
"""

from enum import Enum, IntEnum
from typing import Dict, Final

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Earth's radius in meters (WGS84 mean radius)
EARTH_RADIUS_METERS: Final[float] = 6_371_008.8

# Speed limits (m/s)
class SpeedLimits:
    """Maximum realistic speeds by activity type (m/s)."""
    WALKING: Final[float] = 2.5       # ~9 km/h
    RUNNING: Final[float] = 12.0      # ~43 km/h (elite sprinter)
    CYCLING: Final[float] = 25.0      # ~90 km/h (downhill)
    DRIVING: Final[float] = 50.0      # ~180 km/h (highway)
    HIGH_SPEED_RAIL: Final[float] = 100.0  # ~360 km/h
    AIRCRAFT: Final[float] = 300.0    # ~1080 km/h (commercial jet)
    
    # Default maximum for unknown activity
    DEFAULT_MAX: Final[float] = 50.0


# Acceleration limits (m/s²)
class AccelerationLimits:
    """Maximum realistic acceleration by context."""
    HUMAN_COMFORTABLE: Final[float] = 3.0    # ~0.3g
    VEHICLE_NORMAL: Final[float] = 5.0       # ~0.5g
    VEHICLE_EMERGENCY: Final[float] = 10.0   # ~1.0g
    MAXIMUM_REALISTIC: Final[float] = 15.0   # ~1.5g


# =============================================================================
# GPS ACCURACY THRESHOLDS
# =============================================================================

class GPSAccuracyTiers:
    """GPS accuracy tiers in meters."""
    EXCELLENT: Final[float] = 5.0      # Indoor GPS, RTK
    GOOD: Final[float] = 10.0          # Clear sky, good conditions
    MODERATE: Final[float] = 25.0      # Urban canyon, some obstruction
    POOR: Final[float] = 50.0          # Heavy obstruction
    UNRELIABLE: Final[float] = 100.0   # Indoor, tunnel, severe conditions


# =============================================================================
# ACTIVITY TYPES
# =============================================================================

class ActivityType(str, Enum):
    """
    Activity types recognized by the mobile app.
    
    These come from device motion sensors and are used
    for context-aware validation.
    """
    STILL = "still"
    WALKING = "walking"
    ON_FOOT = "on_foot"
    RUNNING = "running"
    ON_BICYCLE = "on_bicycle"
    IN_VEHICLE = "in_vehicle"
    SHAKING = "shaking"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_string(cls, value: str) -> "ActivityType":
        """Parse activity type from string, defaulting to UNKNOWN."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.UNKNOWN
    
    @property
    def max_speed_ms(self) -> float:
        """Get maximum realistic speed for this activity type."""
        speed_map: Dict[str, float] = {
            "still": 0.5,  # Small drift allowed
            "walking": SpeedLimits.WALKING,
            "on_foot": SpeedLimits.WALKING,
            "running": SpeedLimits.RUNNING,
            "on_bicycle": SpeedLimits.CYCLING,
            "in_vehicle": SpeedLimits.DRIVING,
            "shaking": 1.0,  # Minimal movement
            "unknown": SpeedLimits.DEFAULT_MAX,
        }
        return speed_map.get(self.value, SpeedLimits.DEFAULT_MAX)


class ActivityConfidence(IntEnum):
    """
    Activity detection confidence levels.
    
    Values from device motion APIs (iOS CoreMotion, Android Activity Recognition).
    """
    LOW = 0
    MEDIUM_LOW = 33
    MEDIUM = 66
    MEDIUM_HIGH = 80
    HIGH = 100


# =============================================================================
# DATA QUALITY FLAGS
# =============================================================================

class DataQualityFlag(str, Enum):
    """
    Flags indicating specific data quality issues.
    
    These are not errors but observations that affect confidence scoring.
    Multiple flags can apply to a single data point.
    """
    # Coordinate Issues
    INVALID_COORDINATES = "invalid_coordinates"
    UNREALISTIC_SPEED = "unrealistic_speed"
    UNREALISTIC_ACCELERATION = "unrealistic_acceleration"
    TELEPORTATION = "teleportation"  # Impossible distance in time
    
    # GPS Issues
    LOW_GPS_ACCURACY = "low_gps_accuracy"
    GPS_DRIFT = "gps_drift"  # Detected stationary drift
    
    # Timing Issues
    DUPLICATE_TIMESTAMP = "duplicate_timestamp"
    TIMESTAMP_OUT_OF_ORDER = "timestamp_out_of_order"
    LARGE_TIME_GAP = "large_time_gap"
    RAPID_UPDATES = "rapid_updates"  # Faster than physically possible
    
    # Consistency Issues
    SPEED_ACTIVITY_MISMATCH = "speed_activity_mismatch"
    MOVEMENT_STILLNESS_CONFLICT = "movement_stillness_conflict"
    
    # Data Quality
    MISSING_SPEED = "missing_speed"
    MISSING_ACCURACY = "missing_accuracy"
    INTERPOLATED = "interpolated"


# =============================================================================
# CONFIDENCE SCORE WEIGHTS
# =============================================================================

class ConfidenceWeights:
    """
    Weights for confidence score calculation.
    
    These weights determine how much each factor contributes
    to the overall confidence score. Sum should equal 1.0.
    """
    GPS_ACCURACY: Final[float] = 0.25
    SPEED_VALIDITY: Final[float] = 0.20
    ACCELERATION_VALIDITY: Final[float] = 0.15
    TEMPORAL_CONSISTENCY: Final[float] = 0.15
    ACTIVITY_CONSISTENCY: Final[float] = 0.15
    SIGNAL_CONTINUITY: Final[float] = 0.10


# =============================================================================
# TIME CONSTANTS
# =============================================================================

class TimeConstants:
    """Time-related constants for data processing."""
    # Minimum time between valid updates (seconds)
    MIN_UPDATE_INTERVAL: Final[float] = 0.5
    
    # Maximum gap before flagging (seconds)
    MAX_NORMAL_GAP: Final[float] = 30.0
    MAX_ACCEPTABLE_GAP: Final[float] = 300.0  # 5 minutes
    
    # Stop detection threshold (seconds)
    MIN_STOP_DURATION: Final[float] = 60.0  # 1 minute
    
    # Timestamp format for parsing
    TIMESTAMP_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


# =============================================================================
# BATCH PROCESSING
# =============================================================================

class BatchConstants:
    """Constants for batch processing operations."""
    DEFAULT_BATCH_SIZE: Final[int] = 1000
    MAX_BATCH_SIZE: Final[int] = 10000
    STREAM_CHUNK_SIZE: Final[int] = 100


# =============================================================================
# API CONSTANTS
# =============================================================================

class APIConstants:
    """API-related constants."""
    DEFAULT_PAGE_SIZE: Final[int] = 100
    MAX_PAGE_SIZE: Final[int] = 1000
    API_PREFIX: Final[str] = "/api/v1"
