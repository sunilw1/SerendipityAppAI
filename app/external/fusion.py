"""
Travel Context Fusion
======================

Combines weather and traffic data into unified travel context.

Features:
- Weighted combination of weather and traffic impacts
- Confidence scoring based on data availability
- Time-aware context validity
- Explainable delay factors

Design Principles:
- Graceful degradation when data sources unavailable
- Clear factor attribution
- Conservative delay estimates (err on side of caution)
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple
import uuid

from app.core.logging import get_logger
from app.models.phase3_schemas import (
    WeatherData,
    TrafficData,
    TravelContext,
    TrafficSeverity,
    WeatherCondition,
)
from app.external.weather import get_weather_client, WeatherClient
from app.external.traffic import get_traffic_client, TrafficClient

logger = get_logger(__name__)

# Weight factors for combining impacts
WEATHER_WEIGHT = 0.35
TRAFFIC_WEIGHT = 0.65

# Context validity duration
CONTEXT_VALIDITY_MINUTES = 15


class TravelContextFusion:
    """
    Fuses weather and traffic data into unified travel context.
    
    Used by predictive routing service.
    """
    
    def __init__(
        self,
        weather_client: Optional[WeatherClient] = None,
        traffic_client: Optional[TrafficClient] = None,
    ):
        """
        Initialize fusion service.
        
        Args:
            weather_client: Weather client instance
            traffic_client: Traffic client instance
        """
        self.weather_client = weather_client or get_weather_client()
        self.traffic_client = traffic_client or get_traffic_client()
    
    def _calculate_combined_delay(
        self,
        weather: Optional[WeatherData],
        traffic: Optional[TrafficData],
    ) -> Tuple[float, List[str]]:
        """
        Calculate combined delay factor from weather and traffic.
        
        Args:
            weather: Weather data (optional)
            traffic: Traffic data (optional)
            
        Returns:
            Tuple of (delay_factor, explanations)
        """
        factors = []
        delay_factor = 1.0
        
        # Traffic contribution (primary factor)
        if traffic:
            traffic_delay = traffic.delay_factor
            delay_factor *= (1.0 + (traffic_delay - 1.0) * TRAFFIC_WEIGHT)
            
            if traffic.severity in [TrafficSeverity.HEAVY, TrafficSeverity.SEVERE]:
                factors.append(
                    f"{traffic.severity.value.replace('_', ' ').title()} traffic: "
                    f"{traffic.current_speed_kmh:.0f} km/h "
                    f"(normal: {traffic.free_flow_speed_kmh:.0f} km/h)"
                )
            elif traffic.severity == TrafficSeverity.MODERATE:
                factors.append(
                    f"Moderate traffic congestion detected"
                )
        
        # Weather contribution (secondary factor)
        if weather:
            weather_delay = 1.0 + weather.travel_impact_score
            delay_factor *= (1.0 + (weather_delay - 1.0) * WEATHER_WEIGHT)
            
            if weather.travel_impact_score > 0.3:
                factors.append(
                    f"Weather impact: {weather.condition.value} conditions "
                    f"(visibility: {weather.visibility_meters:.0f}m)"
                )
            elif weather.travel_impact_score > 0.1:
                factors.append(
                    f"Minor weather impact: {weather.condition.value}"
                )
        
        # No data case
        if not factors:
            factors.append("No significant delays expected")
        
        return delay_factor, factors
    
    def _calculate_travel_risk(
        self,
        weather: Optional[WeatherData],
        traffic: Optional[TrafficData],
    ) -> float:
        """
        Calculate overall travel risk score.
        
        Args:
            weather: Weather data (optional)
            traffic: Traffic data (optional)
            
        Returns:
            Risk score (0-1)
        """
        risk = 0.0
        
        # Weather risk
        if weather:
            risk += weather.travel_impact_score * 0.4
        
        # Traffic risk (based on severity)
        if traffic:
            severity_risk = {
                TrafficSeverity.FREE_FLOW: 0.0,
                TrafficSeverity.LIGHT: 0.1,
                TrafficSeverity.MODERATE: 0.3,
                TrafficSeverity.HEAVY: 0.6,
                TrafficSeverity.SEVERE: 0.9,
                TrafficSeverity.UNKNOWN: 0.2,
            }
            risk += severity_risk.get(traffic.severity, 0.2) * 0.6
        
        return min(1.0, risk)
    
    def _calculate_confidence(
        self,
        weather_available: bool,
        traffic_available: bool,
        weather_age_seconds: float = 0,
        traffic_age_seconds: float = 0,
    ) -> float:
        """
        Calculate confidence in the travel context.
        
        Args:
            weather_available: Whether weather data is available
            traffic_available: Whether traffic data is available
            weather_age_seconds: Age of weather data
            traffic_age_seconds: Age of traffic data
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence from data availability
        if weather_available and traffic_available:
            base_confidence = 0.95
        elif traffic_available:
            base_confidence = 0.75
        elif weather_available:
            base_confidence = 0.5
        else:
            base_confidence = 0.2
        
        # Decay based on data age
        max_age = 900  # 15 minutes
        avg_age = (weather_age_seconds + traffic_age_seconds) / 2
        age_factor = max(0.5, 1.0 - (avg_age / max_age) * 0.5)
        
        return base_confidence * age_factor
    
    async def get_context(
        self,
        lat: float,
        lon: float,
        destination_lat: Optional[float] = None,
        destination_lon: Optional[float] = None,
    ) -> TravelContext:
        """
        Get travel context for a location or route.
        
        Args:
            lat: Origin latitude
            lon: Origin longitude
            destination_lat: Destination latitude (optional)
            destination_lon: Destination longitude (optional)
            
        Returns:
            TravelContext with fused data
        """
        now = datetime.now(timezone.utc)
        
        # Fetch weather data
        weather = await self.weather_client.get_weather(lat, lon)
        weather_available = weather is not None
        weather_age = 0.0
        if weather:
            weather_age = (now - weather.timestamp).total_seconds()
        
        # Fetch traffic data
        traffic = None
        if destination_lat is not None and destination_lon is not None:
            traffic = await self.traffic_client.get_traffic_for_segment(
                lat, lon, destination_lat, destination_lon
            )
        else:
            traffic = await self.traffic_client.get_traffic_flow(lat, lon)
        
        traffic_available = traffic is not None
        traffic_age = 0.0
        if traffic:
            traffic_age = (now - traffic.timestamp).total_seconds()
        
        # Calculate combined metrics
        delay_factor, factors = self._calculate_combined_delay(weather, traffic)
        travel_risk = self._calculate_travel_risk(weather, traffic)
        confidence = self._calculate_confidence(
            weather_available, traffic_available,
            weather_age, traffic_age
        )
        
        # Validity window
        valid_until = now + timedelta(minutes=CONTEXT_VALIDITY_MINUTES)
        
        context = TravelContext(
            weather=weather,
            weather_available=weather_available,
            traffic=traffic,
            traffic_available=traffic_available,
            combined_delay_factor=delay_factor,
            travel_risk_score=travel_risk,
            context_confidence=confidence,
            factors=factors,
            generated_at=now,
            valid_until=valid_until,
        )
        
        logger.info(
            "travel_context_generated",
            lat=lat,
            lon=lon,
            delay_factor=delay_factor,
            risk=travel_risk,
            confidence=confidence,
        )
        
        return context
    
    async def get_route_context(
        self,
        waypoints: List[Tuple[float, float]],
    ) -> TravelContext:
        """
        Get aggregated travel context for a route.
        
        Args:
            waypoints: List of (lat, lon) tuples
            
        Returns:
            Aggregated TravelContext
        """
        if not waypoints:
            return await self.get_context(0, 0)
        
        if len(waypoints) == 1:
            return await self.get_context(waypoints[0][0], waypoints[0][1])
        
        # Get context for start and end
        start = waypoints[0]
        end = waypoints[-1]
        
        # Use midpoint for weather (weather is more regional)
        mid_lat = (start[0] + end[0]) / 2
        mid_lon = (start[1] + end[1]) / 2
        
        now = datetime.now(timezone.utc)
        
        # Fetch weather for midpoint
        weather = await self.weather_client.get_weather(mid_lat, mid_lon)
        weather_available = weather is not None
        
        # Fetch traffic for the route
        total_delay, avg_delay_factor, traffic_explanations = (
            await self.traffic_client.get_route_delay_estimate(waypoints)
        )
        
        # Create synthetic traffic data representing the whole route
        traffic = TrafficData(
            severity=self._severity_from_delay_factor(avg_delay_factor),
            current_speed_kmh=50.0 / avg_delay_factor,  # Approximate
            free_flow_speed_kmh=50.0,
            congestion_ratio=1.0 / avg_delay_factor,
            delay_factor=avg_delay_factor,
            estimated_delay_seconds=total_delay,
            segment_start=start,
            segment_end=end,
            segment_length_meters=self._estimate_route_length(waypoints),
            timestamp=now,
            source="tomtom_aggregated",
        )
        
        traffic_available = True
        
        # Calculate combined metrics
        delay_factor, factors = self._calculate_combined_delay(weather, traffic)
        
        # Add traffic explanations
        factors.extend(traffic_explanations)
        
        travel_risk = self._calculate_travel_risk(weather, traffic)
        confidence = self._calculate_confidence(
            weather_available, traffic_available, 0, 0
        )
        
        valid_until = now + timedelta(minutes=CONTEXT_VALIDITY_MINUTES)
        
        return TravelContext(
            weather=weather,
            weather_available=weather_available,
            traffic=traffic,
            traffic_available=traffic_available,
            combined_delay_factor=delay_factor,
            travel_risk_score=travel_risk,
            context_confidence=confidence,
            factors=factors,
            generated_at=now,
            valid_until=valid_until,
        )
    
    def _severity_from_delay_factor(self, delay_factor: float) -> TrafficSeverity:
        """Convert delay factor to traffic severity."""
        if delay_factor >= 3.0:
            return TrafficSeverity.SEVERE
        elif delay_factor >= 2.0:
            return TrafficSeverity.HEAVY
        elif delay_factor >= 1.5:
            return TrafficSeverity.MODERATE
        elif delay_factor >= 1.2:
            return TrafficSeverity.LIGHT
        else:
            return TrafficSeverity.FREE_FLOW
    
    def _estimate_route_length(
        self,
        waypoints: List[Tuple[float, float]],
    ) -> float:
        """Estimate total route length in meters."""
        from app.utils.geo import haversine_distance
        
        total = 0.0
        for i in range(len(waypoints) - 1):
            total += haversine_distance(
                waypoints[i][0], waypoints[i][1],
                waypoints[i+1][0], waypoints[i+1][1],
            )
        return total


# Singleton instance
_fusion_service: Optional[TravelContextFusion] = None


def get_travel_context_fusion() -> TravelContextFusion:
    """Get or create singleton fusion service."""
    global _fusion_service
    if _fusion_service is None:
        _fusion_service = TravelContextFusion()
    return _fusion_service


async def get_travel_context(
    lat: float,
    lon: float,
    destination_lat: Optional[float] = None,
    destination_lon: Optional[float] = None,
) -> TravelContext:
    """
    Convenience function to get travel context.
    
    Args:
        lat: Origin latitude
        lon: Origin longitude
        destination_lat: Destination latitude (optional)
        destination_lon: Destination longitude (optional)
        
    Returns:
        TravelContext
    """
    service = get_travel_context_fusion()
    return await service.get_context(lat, lon, destination_lat, destination_lon)
