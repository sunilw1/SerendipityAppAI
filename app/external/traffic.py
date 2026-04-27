"""
TomTom Traffic Integration
===========================

Async client for TomTom Traffic Flow API with caching.

Features:
- Real-time traffic flow data
- Delay factor calculation
- Route segment analysis
- Redis caching with TTL

Design Principles:
- Non-blocking async operations
- Cache-first approach
- Graceful degradation on API failures
- Clear traffic severity classification
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple, List
import hashlib

import httpx

from app.core.logging import get_logger
from app.core.config import settings
from app.models.phase3_schemas import TrafficData, TrafficSeverity

logger = get_logger(__name__)

# Traffic congestion ratio to severity mapping
CONGESTION_SEVERITY_THRESHOLDS = [
    (0.0, 0.2, TrafficSeverity.SEVERE),      # < 20% of free flow
    (0.2, 0.4, TrafficSeverity.HEAVY),       # 20-40% of free flow
    (0.4, 0.6, TrafficSeverity.MODERATE),    # 40-60% of free flow
    (0.6, 0.85, TrafficSeverity.LIGHT),      # 60-85% of free flow
    (0.85, 1.0, TrafficSeverity.FREE_FLOW),  # > 85% of free flow
]


def get_traffic_severity(congestion_ratio: float) -> TrafficSeverity:
    """
    Convert congestion ratio to traffic severity.
    
    Args:
        congestion_ratio: Current speed / free flow speed (0-1)
        
    Returns:
        TrafficSeverity enum
    """
    for min_val, max_val, severity in CONGESTION_SEVERITY_THRESHOLDS:
        if min_val <= congestion_ratio < max_val:
            return severity
    return TrafficSeverity.FREE_FLOW


def calculate_delay_factor(congestion_ratio: float) -> float:
    """
    Calculate delay factor from congestion ratio.
    
    Args:
        congestion_ratio: Current speed / free flow speed (0-1)
        
    Returns:
        Delay factor (1.0 = no delay, higher = more delay)
    """
    if congestion_ratio <= 0:
        return 5.0  # Maximum delay factor for standstill
    
    # Delay factor is inverse of congestion ratio
    # But capped to avoid extreme values
    delay = 1.0 / congestion_ratio
    return min(5.0, max(1.0, delay))


class TrafficCache:
    """
    In-memory cache for traffic data.
    
    Traffic data changes frequently, so shorter TTL than weather.
    In production, replace with Redis.
    """
    
    def __init__(self, ttl_seconds: int = 300):  # 5 minutes default
        """
        Initialize cache.
        
        Args:
            ttl_seconds: Cache TTL in seconds
        """
        self.ttl = ttl_seconds
        self._cache: Dict[str, Tuple[TrafficData, datetime]] = {}
    
    def _make_key(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
    ) -> str:
        """Create cache key from segment endpoints."""
        # Round to 3 decimal places for traffic granularity
        segment = f"{start_lat:.3f},{start_lon:.3f}-{end_lat:.3f},{end_lon:.3f}"
        return hashlib.md5(segment.encode()).hexdigest()
    
    def get(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
    ) -> Optional[TrafficData]:
        """Get cached traffic data if valid."""
        key = self._make_key(start_lat, start_lon, end_lat, end_lon)
        
        if key not in self._cache:
            return None
        
        data, cached_at = self._cache[key]
        
        if datetime.now(timezone.utc) - cached_at > timedelta(seconds=self.ttl):
            del self._cache[key]
            return None
        
        return data
    
    def set(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        data: TrafficData,
    ) -> None:
        """Cache traffic data."""
        key = self._make_key(start_lat, start_lon, end_lat, end_lon)
        self._cache[key] = (data, datetime.now(timezone.utc))
    
    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()


class TrafficClient:
    """
    Async client for TomTom Traffic Flow API.
    
    Provides real-time traffic data with delay calculations.
    """
    
    # TomTom Traffic Flow API endpoint
    BASE_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl_seconds: int = 300,
        timeout_seconds: float = 10.0,
    ):
        """
        Initialize traffic client.
        
        Args:
            api_key: TomTom API key
            cache_ttl_seconds: Cache TTL
            timeout_seconds: Request timeout
        """
        self.api_key = api_key or getattr(settings, 'tomtom_api_key', None)
        self.timeout = timeout_seconds
        self.cache = TrafficCache(ttl_seconds=cache_ttl_seconds)
        self._client: Optional[httpx.AsyncClient] = None
        
        if not self.api_key:
            logger.warning(
                "traffic_client_no_api_key",
                note="TomTom API key not configured"
            )
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def get_traffic_flow(
        self,
        lat: float,
        lon: float,
        use_cache: bool = True,
    ) -> Optional[TrafficData]:
        """
        Get traffic flow data for a point.
        
        TomTom returns traffic for the road segment nearest to the point.
        
        Args:
            lat: Latitude
            lon: Longitude
            use_cache: Whether to use cache
            
        Returns:
            TrafficData or None if unavailable
        """
        # For single point queries, use same coords for start/end
        return await self.get_traffic_for_segment(lat, lon, lat, lon, use_cache)
    
    async def get_traffic_for_segment(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        use_cache: bool = True,
    ) -> Optional[TrafficData]:
        """
        Get traffic flow data for a road segment.
        
        Args:
            start_lat: Start latitude
            start_lon: Start longitude
            end_lat: End latitude
            end_lon: End longitude
            use_cache: Whether to use cache
            
        Returns:
            TrafficData or None if unavailable
        """
        # Check cache first
        if use_cache:
            cached = self.cache.get(start_lat, start_lon, end_lat, end_lon)
            if cached:
                logger.debug(
                    "traffic_cache_hit",
                    start=(start_lat, start_lon),
                    end=(end_lat, end_lon),
                )
                return cached
        
        # No API key - return None
        if not self.api_key:
            logger.warning("traffic_api_key_missing")
            return None
        
        try:
            client = await self._get_client()
            
            # Use midpoint for single point query
            mid_lat = (start_lat + end_lat) / 2
            mid_lon = (start_lon + end_lon) / 2
            
            params = {
                "key": self.api_key,
                "point": f"{mid_lat},{mid_lon}",
                "unit": "KMPH",
                "thickness": 10,
            }
            
            response = await client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse flow segment data
            flow_data = data.get("flowSegmentData", {})
            
            current_speed = flow_data.get("currentSpeed", 0)
            free_flow_speed = flow_data.get("freeFlowSpeed", 1)
            
            # Avoid division by zero
            if free_flow_speed <= 0:
                free_flow_speed = 1
            
            congestion_ratio = current_speed / free_flow_speed
            congestion_ratio = min(1.0, max(0.0, congestion_ratio))
            
            severity = get_traffic_severity(congestion_ratio)
            delay_factor = calculate_delay_factor(congestion_ratio)
            
            # Estimate segment length (rough calculation)
            from app.utils.geo import haversine_distance
            segment_length = haversine_distance(
                start_lat, start_lon, end_lat, end_lon
            )
            
            # Estimate delay in seconds
            if current_speed > 0:
                actual_time = (segment_length / 1000) / current_speed * 3600
                free_flow_time = (segment_length / 1000) / free_flow_speed * 3600
                estimated_delay = max(0, actual_time - free_flow_time)
            else:
                estimated_delay = 0
            
            now = datetime.now(timezone.utc)
            
            traffic_data = TrafficData(
                severity=severity,
                current_speed_kmh=float(current_speed),
                free_flow_speed_kmh=float(free_flow_speed),
                congestion_ratio=congestion_ratio,
                delay_factor=delay_factor,
                estimated_delay_seconds=estimated_delay,
                segment_start=(start_lat, start_lon),
                segment_end=(end_lat, end_lon),
                segment_length_meters=segment_length,
                timestamp=now,
                source="tomtom",
            )
            
            # Cache the result
            self.cache.set(start_lat, start_lon, end_lat, end_lon, traffic_data)
            
            logger.info(
                "traffic_fetched",
                start=(start_lat, start_lon),
                severity=severity.value,
                delay_factor=delay_factor,
            )
            
            return traffic_data
            
        except httpx.HTTPStatusError as e:
            logger.error(
                "traffic_api_error",
                status=e.response.status_code,
                lat=start_lat,
                lon=start_lon,
            )
            return None
            
        except Exception as e:
            logger.error(
                "traffic_fetch_failed",
                error=str(e),
                lat=start_lat,
                lon=start_lon,
            )
            return None
    
    async def get_traffic_for_route(
        self,
        waypoints: List[Tuple[float, float]],
    ) -> List[Optional[TrafficData]]:
        """
        Get traffic for segments along a route.
        
        Args:
            waypoints: List of (lat, lon) tuples defining route
            
        Returns:
            List of TrafficData for each segment
        """
        if len(waypoints) < 2:
            return []
        
        results = []
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            
            traffic = await self.get_traffic_for_segment(
                start[0], start[1],
                end[0], end[1],
            )
            results.append(traffic)
        
        return results
    
    async def get_route_delay_estimate(
        self,
        waypoints: List[Tuple[float, float]],
    ) -> Tuple[float, float, List[str]]:
        """
        Get total delay estimate for a route.
        
        Args:
            waypoints: List of (lat, lon) tuples
            
        Returns:
            Tuple of (total_delay_seconds, avg_delay_factor, explanations)
        """
        traffic_data = await self.get_traffic_for_route(waypoints)
        
        total_delay = 0.0
        delay_factors = []
        explanations = []
        
        for i, data in enumerate(traffic_data):
            if data:
                total_delay += data.estimated_delay_seconds
                delay_factors.append(data.delay_factor)
                
                if data.severity in [TrafficSeverity.HEAVY, TrafficSeverity.SEVERE]:
                    explanations.append(
                        f"Segment {i+1}: {data.severity.value} traffic "
                        f"({data.current_speed_kmh:.0f} km/h vs "
                        f"{data.free_flow_speed_kmh:.0f} km/h free flow)"
                    )
        
        avg_delay_factor = (
            sum(delay_factors) / len(delay_factors)
            if delay_factors else 1.0
        )
        
        return total_delay, avg_delay_factor, explanations


# Singleton instance
_traffic_client: Optional[TrafficClient] = None


def get_traffic_client() -> TrafficClient:
    """Get or create singleton traffic client."""
    global _traffic_client
    if _traffic_client is None:
        _traffic_client = TrafficClient()
    return _traffic_client


async def get_traffic_for_location(
    lat: float,
    lon: float,
) -> Optional[TrafficData]:
    """
    Convenience function to get traffic for a location.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        TrafficData or None
    """
    client = get_traffic_client()
    return await client.get_traffic_flow(lat, lon)
