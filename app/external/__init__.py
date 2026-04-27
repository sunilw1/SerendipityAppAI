"""
External Data Integration
=========================

Phase 3 external data sources for predictive intelligence.

Modules:
- weather: OpenWeatherMap API integration
- traffic: TomTom Traffic API integration
- fusion: Weather + traffic data fusion
"""

from app.external.weather import WeatherClient, get_weather_client
from app.external.traffic import TrafficClient, get_traffic_client
from app.external.fusion import TravelContextFusion, get_travel_context

__all__ = [
    "WeatherClient",
    "get_weather_client",
    "TrafficClient",
    "get_traffic_client",
    "TravelContextFusion",
    "get_travel_context",
]
