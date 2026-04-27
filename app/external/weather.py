"""
OpenWeatherMap Integration
===========================

Async client for OpenWeatherMap API with caching.

Features:
- Async HTTP requests
- Redis caching with TTL
- Weather condition to travel impact mapping
- Graceful fallback on API failures

Design Principles:
- Non-blocking async operations
- Cache-first approach for efficiency
- Clear error handling
- Explainable impact scoring
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, Tuple
from functools import lru_cache
import hashlib

import httpx

from app.core.logging import get_logger
from app.core.config import settings
from app.models.phase3_schemas import WeatherData, WeatherCondition

logger = get_logger(__name__)

# Weather condition to travel impact mapping
WEATHER_IMPACT_MAP: Dict[str, Tuple[WeatherCondition, float]] = {
    # Clear conditions - minimal impact
    "clear": (WeatherCondition.CLEAR, 0.0),
    "clear sky": (WeatherCondition.CLEAR, 0.0),
    
    # Cloudy conditions - very low impact
    "few clouds": (WeatherCondition.CLOUDS, 0.05),
    "scattered clouds": (WeatherCondition.CLOUDS, 0.05),
    "broken clouds": (WeatherCondition.CLOUDS, 0.1),
    "overcast clouds": (WeatherCondition.CLOUDS, 0.1),
    
    # Light precipitation - moderate impact
    "light rain": (WeatherCondition.RAIN, 0.3),
    "moderate rain": (WeatherCondition.RAIN, 0.4),
    "light intensity drizzle": (WeatherCondition.DRIZZLE, 0.2),
    "drizzle": (WeatherCondition.DRIZZLE, 0.25),
    
    # Heavy precipitation - high impact
    "heavy intensity rain": (WeatherCondition.RAIN, 0.6),
    "very heavy rain": (WeatherCondition.RAIN, 0.7),
    "extreme rain": (WeatherCondition.RAIN, 0.85),
    "freezing rain": (WeatherCondition.RAIN, 0.8),
    "shower rain": (WeatherCondition.RAIN, 0.5),
    
    # Thunderstorms - very high impact
    "thunderstorm": (WeatherCondition.THUNDERSTORM, 0.7),
    "thunderstorm with light rain": (WeatherCondition.THUNDERSTORM, 0.65),
    "thunderstorm with rain": (WeatherCondition.THUNDERSTORM, 0.75),
    "thunderstorm with heavy rain": (WeatherCondition.THUNDERSTORM, 0.85),
    "heavy thunderstorm": (WeatherCondition.THUNDERSTORM, 0.9),
    
    # Snow - high impact
    "light snow": (WeatherCondition.SNOW, 0.5),
    "snow": (WeatherCondition.SNOW, 0.65),
    "heavy snow": (WeatherCondition.SNOW, 0.8),
    "sleet": (WeatherCondition.SNOW, 0.7),
    
    # Visibility issues - moderate to high impact
    "mist": (WeatherCondition.MIST, 0.3),
    "fog": (WeatherCondition.FOG, 0.5),
    "haze": (WeatherCondition.HAZE, 0.2),
    "smoke": (WeatherCondition.HAZE, 0.4),
    "dust": (WeatherCondition.HAZE, 0.35),
    "sand": (WeatherCondition.HAZE, 0.4),
}

# Default impact for unknown conditions
DEFAULT_WEATHER_IMPACT = 0.2


class WeatherCache:
    """
    In-memory cache for weather data.
    
    In production, replace with Redis for distributed caching.
    """
    
    def __init__(self, ttl_seconds: int = 900):  # 15 minutes default
        """
        Initialize cache.
        
        Args:
            ttl_seconds: Cache TTL in seconds
        """
        self.ttl = ttl_seconds
        self._cache: Dict[str, Tuple[WeatherData, datetime]] = {}
    
    def _make_key(self, lat: float, lon: float) -> str:
        """Create cache key from coordinates."""
        # Round to 2 decimal places for reasonable granularity
        rounded = f"{lat:.2f},{lon:.2f}"
        return hashlib.md5(rounded.encode()).hexdigest()
    
    def get(self, lat: float, lon: float) -> Optional[WeatherData]:
        """Get cached weather data if valid."""
        key = self._make_key(lat, lon)
        
        if key not in self._cache:
            return None
        
        data, cached_at = self._cache[key]
        
        # Check if expired
        if datetime.now(timezone.utc) - cached_at > timedelta(seconds=self.ttl):
            del self._cache[key]
            return None
        
        return data
    
    def set(self, lat: float, lon: float, data: WeatherData) -> None:
        """Cache weather data."""
        key = self._make_key(lat, lon)
        self._cache[key] = (data, datetime.now(timezone.utc))
    
    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()


class WeatherClient:
    """
    Async client for OpenWeatherMap API.
    
    Provides weather data with travel impact scoring.
    """
    
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl_seconds: int = 900,
        timeout_seconds: float = 10.0,
    ):
        """
        Initialize weather client.
        
        Args:
            api_key: OpenWeatherMap API key
            cache_ttl_seconds: Cache TTL
            timeout_seconds: Request timeout
        """
        self.api_key = api_key or getattr(settings, 'openweathermap_api_key', None)
        self.timeout = timeout_seconds
        self.cache = WeatherCache(ttl_seconds=cache_ttl_seconds)
        self._client: Optional[httpx.AsyncClient] = None
        
        if not self.api_key:
            logger.warning(
                "weather_client_no_api_key",
                note="OpenWeatherMap API key not configured"
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
    
    def _calculate_travel_impact(
        self,
        condition: str,
        wind_speed: float,
        visibility: float,
        precipitation: float,
    ) -> Tuple[WeatherCondition, float]:
        """
        Calculate travel impact from weather conditions.
        
        Args:
            condition: Weather condition string
            wind_speed: Wind speed in m/s
            visibility: Visibility in meters
            precipitation: Precipitation in mm
            
        Returns:
            Tuple of (WeatherCondition, impact_score)
        """
        # Get base impact from condition
        condition_lower = condition.lower()
        weather_type, base_impact = WEATHER_IMPACT_MAP.get(
            condition_lower,
            (WeatherCondition.UNKNOWN, DEFAULT_WEATHER_IMPACT)
        )
        
        # Adjust for wind speed (high winds add impact)
        wind_impact = 0.0
        if wind_speed > 15:  # Strong wind
            wind_impact = 0.2
        elif wind_speed > 10:  # Moderate wind
            wind_impact = 0.1
        elif wind_speed > 7:  # Light wind
            wind_impact = 0.05
        
        # Adjust for visibility (low visibility adds impact)
        visibility_impact = 0.0
        if visibility < 200:  # Very poor
            visibility_impact = 0.4
        elif visibility < 500:  # Poor
            visibility_impact = 0.25
        elif visibility < 1000:  # Moderate
            visibility_impact = 0.15
        elif visibility < 2000:  # Reduced
            visibility_impact = 0.05
        
        # Combine impacts (capped at 1.0)
        total_impact = min(1.0, base_impact + wind_impact + visibility_impact)
        
        return weather_type, total_impact
    
    async def get_weather(
        self,
        lat: float,
        lon: float,
        use_cache: bool = True,
    ) -> Optional[WeatherData]:
        """
        Get weather data for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            use_cache: Whether to use cache
            
        Returns:
            WeatherData or None if unavailable
        """
        # Check cache first
        if use_cache:
            cached = self.cache.get(lat, lon)
            if cached:
                logger.debug("weather_cache_hit", lat=lat, lon=lon)
                return cached
        
        # No API key - return None
        if not self.api_key:
            logger.warning("weather_api_key_missing")
            return None
        
        try:
            client = await self._get_client()
            
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric",
            }
            
            response = await client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse response
            weather_main = data.get("weather", [{}])[0]
            main = data.get("main", {})
            wind = data.get("wind", {})
            
            # Get precipitation (rain or snow in last hour)
            rain = data.get("rain", {}).get("1h", 0.0)
            snow = data.get("snow", {}).get("1h", 0.0)
            precipitation = rain + snow
            
            visibility = data.get("visibility", 10000)  # Default 10km
            
            # Calculate travel impact
            condition_desc = weather_main.get("description", "unknown")
            wind_speed = wind.get("speed", 0.0)
            
            weather_condition, travel_impact = self._calculate_travel_impact(
                condition_desc,
                wind_speed,
                visibility,
                precipitation,
            )
            
            now = datetime.now(timezone.utc)
            
            weather_data = WeatherData(
                condition=weather_condition,
                temperature_celsius=main.get("temp", 20.0),
                humidity_percent=main.get("humidity", 50.0),
                wind_speed_ms=wind_speed,
                visibility_meters=float(visibility),
                precipitation_mm=precipitation,
                travel_impact_score=travel_impact,
                timestamp=now,
                location_lat=lat,
                location_lon=lon,
                source="openweathermap",
            )
            
            # Cache the result
            self.cache.set(lat, lon, weather_data)
            
            logger.info(
                "weather_fetched",
                lat=lat,
                lon=lon,
                condition=weather_condition.value,
                impact=travel_impact,
            )
            
            return weather_data
            
        except httpx.HTTPStatusError as e:
            logger.error(
                "weather_api_error",
                status=e.response.status_code,
                lat=lat,
                lon=lon,
            )
            return None
            
        except Exception as e:
            logger.error(
                "weather_fetch_failed",
                error=str(e),
                lat=lat,
                lon=lon,
            )
            return None
    
    async def get_weather_for_route(
        self,
        waypoints: list[Tuple[float, float]],
    ) -> list[Optional[WeatherData]]:
        """
        Get weather for multiple waypoints along a route.
        
        Args:
            waypoints: List of (lat, lon) tuples
            
        Returns:
            List of WeatherData for each waypoint
        """
        results = []
        for lat, lon in waypoints:
            weather = await self.get_weather(lat, lon)
            results.append(weather)
        return results


# Singleton instance
_weather_client: Optional[WeatherClient] = None


def get_weather_client() -> WeatherClient:
    """Get or create singleton weather client."""
    global _weather_client
    if _weather_client is None:
        _weather_client = WeatherClient()
    return _weather_client


async def get_weather_for_location(
    lat: float,
    lon: float,
) -> Optional[WeatherData]:
    """
    Convenience function to get weather for a location.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        WeatherData or None
    """
    client = get_weather_client()
    return await client.get_weather(lat, lon)
