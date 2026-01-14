"""
Geospatial Utilities
=====================

High-performance geographic calculations for location tracking.

Key Functions:
- haversine_distance: Distance between two coordinates
- calculate_bearing: Compass bearing between points
- calculate_speed: Speed from distance and time
- is_within_bounds: Geographic boundary check

Performance Notes:
- All calculations use radians internally
- Vectorized operations available for batch processing
- NumPy used for performance on large datasets
"""

import math
from typing import Tuple, Optional, List
import numpy as np

from app.core.constants import EARTH_RADIUS_METERS


def haversine_distance(
    lat1: float, 
    lon1: float, 
    lat2: float, 
    lon2: float
) -> float:
    """
    Calculate the great-circle distance between two points using Haversine formula.
    
    This is the standard method for GPS distance calculation, accurate
    for distances from centimeters to thousands of kilometers.
    
    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees
        
    Returns:
        Distance in meters
        
    Example:
        >>> haversine_distance(34.0522, -118.2437, 34.0525, -118.2440)
        44.35  # approximately
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = (
        math.sin(delta_lat / 2) ** 2 +
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return EARTH_RADIUS_METERS * c


def haversine_distance_vectorized(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray
) -> np.ndarray:
    """
    Vectorized Haversine distance calculation for batch processing.
    
    Significantly faster for large datasets (100x+ speedup for 10K+ points).
    
    Args:
        lat1: Array of latitudes for first points
        lon1: Array of longitudes for first points
        lat2: Array of latitudes for second points
        lon2: Array of longitudes for second points
        
    Returns:
        Array of distances in meters
    """
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = (
        np.sin(delta_lat / 2) ** 2 +
        np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return EARTH_RADIUS_METERS * c


def calculate_bearing(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculate the initial bearing from point 1 to point 2.
    
    The bearing is the compass direction (0-360 degrees, 0 = North).
    
    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees
        
    Returns:
        Bearing in degrees (0-360, 0 = North, 90 = East)
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon = math.radians(lon2 - lon1)
    
    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = (
        math.cos(lat1_rad) * math.sin(lat2_rad) -
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
    )
    
    bearing_rad = math.atan2(x, y)
    bearing_deg = math.degrees(bearing_rad)
    
    # Normalize to 0-360
    return (bearing_deg + 360) % 360


def calculate_bearing_change(bearing1: float, bearing2: float) -> float:
    """
    Calculate the change in bearing, accounting for 360° wraparound.
    
    Args:
        bearing1: Initial bearing in degrees
        bearing2: Final bearing in degrees
        
    Returns:
        Bearing change in degrees (-180 to 180)
    """
    diff = bearing2 - bearing1
    
    # Normalize to -180 to 180
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
        
    return diff


def calculate_speed(
    distance_meters: float,
    time_seconds: float
) -> Optional[float]:
    """
    Calculate speed from distance and time.
    
    Args:
        distance_meters: Distance traveled in meters
        time_seconds: Time elapsed in seconds
        
    Returns:
        Speed in m/s, or None if time is zero/negative
    """
    if time_seconds <= 0:
        return None
    return distance_meters / time_seconds


def calculate_acceleration(
    speed1_ms: float,
    speed2_ms: float,
    time_seconds: float
) -> Optional[float]:
    """
    Calculate acceleration from speed change.
    
    Args:
        speed1_ms: Initial speed in m/s
        speed2_ms: Final speed in m/s
        time_seconds: Time elapsed in seconds
        
    Returns:
        Acceleration in m/s², or None if time is zero/negative
    """
    if time_seconds <= 0:
        return None
    return (speed2_ms - speed1_ms) / time_seconds


def is_valid_coordinate(lat: float, lon: float) -> bool:
    """
    Check if coordinates are valid (within Earth bounds).
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        
    Returns:
        True if coordinates are valid
    """
    return -90 <= lat <= 90 and -180 <= lon <= 180


def is_null_island(lat: float, lon: float, tolerance: float = 0.0001) -> bool:
    """
    Check if coordinates are at Null Island (0, 0).
    
    Null Island coordinates often indicate GPS initialization errors.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        tolerance: Maximum distance from (0, 0) in degrees
        
    Returns:
        True if coordinates are at or very near Null Island
    """
    return abs(lat) < tolerance and abs(lon) < tolerance


def estimate_gps_error_from_speed(
    reported_speed: float,
    calculated_speed: float,
    time_delta: float
) -> float:
    """
    Estimate GPS error based on speed discrepancy.
    
    Large differences between reported and calculated speed often
    indicate GPS positioning errors.
    
    Args:
        reported_speed: Speed reported by device (m/s)
        calculated_speed: Speed calculated from positions (m/s)
        time_delta: Time between measurements (seconds)
        
    Returns:
        Estimated position error in meters
    """
    speed_diff = abs(reported_speed - calculated_speed)
    return speed_diff * time_delta


def bounding_box(
    lat: float,
    lon: float,
    radius_meters: float
) -> Tuple[float, float, float, float]:
    """
    Calculate a bounding box around a point.
    
    Useful for efficient spatial queries before precise distance checks.
    
    Args:
        lat: Center latitude in degrees
        lon: Center longitude in degrees
        radius_meters: Radius in meters
        
    Returns:
        Tuple of (min_lat, max_lat, min_lon, max_lon)
    """
    # Approximate degrees per meter at this latitude
    lat_rad = math.radians(lat)
    meters_per_lat_degree = 111132.92 - 559.82 * math.cos(2 * lat_rad)
    meters_per_lon_degree = 111412.84 * math.cos(lat_rad)
    
    lat_delta = radius_meters / meters_per_lat_degree
    lon_delta = radius_meters / meters_per_lon_degree
    
    return (
        lat - lat_delta,
        lat + lat_delta,
        lon - lon_delta,
        lon + lon_delta
    )


def simplify_trajectory(
    points: List[Tuple[float, float]],
    epsilon_meters: float = 10.0
) -> List[Tuple[float, float]]:
    """
    Simplify a trajectory using Ramer-Douglas-Peucker algorithm.
    
    Reduces the number of points while preserving the overall shape.
    Useful for visualization and storage optimization.
    
    Args:
        points: List of (lat, lon) tuples
        epsilon_meters: Maximum distance threshold in meters
        
    Returns:
        Simplified list of (lat, lon) tuples
    """
    if len(points) <= 2:
        return points
    
    # Find the point with maximum distance from the line
    max_dist = 0
    max_idx = 0
    
    start = points[0]
    end = points[-1]
    
    for i in range(1, len(points) - 1):
        dist = _perpendicular_distance(points[i], start, end)
        if dist > max_dist:
            max_dist = dist
            max_idx = i
    
    # If max distance is greater than epsilon, recursively simplify
    if max_dist > epsilon_meters:
        left = simplify_trajectory(points[:max_idx + 1], epsilon_meters)
        right = simplify_trajectory(points[max_idx:], epsilon_meters)
        return left[:-1] + right
    else:
        return [start, end]


def _perpendicular_distance(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float]
) -> float:
    """Calculate perpendicular distance from point to line segment."""
    # Approximate using Euclidean distance (good for short segments)
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Convert to approximate meters for the calculation
    lat_mid = (y0 + y1 + y2) / 3
    lat_scale = 111132.92  # meters per degree latitude
    lon_scale = 111412.84 * math.cos(math.radians(lat_mid))  # meters per degree longitude
    
    x0_m, y0_m = x0 * lon_scale, y0 * lat_scale
    x1_m, y1_m = x1 * lon_scale, y1 * lat_scale
    x2_m, y2_m = x2 * lon_scale, y2 * lat_scale
    
    # Line length
    line_len = math.sqrt((x2_m - x1_m) ** 2 + (y2_m - y1_m) ** 2)
    if line_len == 0:
        return math.sqrt((x0_m - x1_m) ** 2 + (y0_m - y1_m) ** 2)
    
    # Calculate perpendicular distance
    dist = abs(
        (y2_m - y1_m) * x0_m - (x2_m - x1_m) * y0_m + x2_m * y1_m - y2_m * x1_m
    ) / line_len
    
    return dist
