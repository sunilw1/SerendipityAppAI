"""
Predictive Routing Service
===========================

Phase 3 service for predicting travel delays and route performance.

Features:
- Historical route pattern matching
- Weather and traffic impact prediction
- LightGBM-based delay forecasting
- GPU-accelerated inference via Triton

Design Principles:
- Uses Phase 2 behavioral profiles
- Explainable predictions
- Confidence-scored outputs
- Battery-safe (backend-only)
"""

from datetime import datetime, timezone
from typing import Optional, List, Tuple, Dict, Any
import uuid
import time

import numpy as np

from app.core.logging import get_logger
from app.core.config import settings
from app.models.phase3_schemas import (
    DelayPrediction,
    TravelContext,
    DelayPredictionRequest,
)
from app.models.phase2_schemas import BehavioralProfile, RoutePattern
from app.external.fusion import TravelContextFusion, get_travel_context_fusion
from app.utils.geo import haversine_distance

logger = get_logger(__name__)

# Default speeds for different conditions (km/h)
DEFAULT_URBAN_SPEED_KMH = 30.0
DEFAULT_HIGHWAY_SPEED_KMH = 80.0
DEFAULT_MIXED_SPEED_KMH = 50.0


class DelayPredictor:
    """
    Predicts travel delays using ML model.
    
    Uses LightGBM model optimized for TensorRT inference.
    Falls back to heuristic prediction when model unavailable.
    """
    
    def __init__(self):
        """Initialize delay predictor."""
        self._model = None
        self._model_version = "1.0"
        self._model_loaded = False
        self._use_gpu = False
        
        # Feature names for the model
        self.feature_names = [
            "distance_meters",
            "historical_duration_seconds",
            "weather_impact",
            "traffic_delay_factor",
            "hour_of_day",
            "day_of_week",
            "is_rush_hour",
            "route_confidence",
        ]
    
    async def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load the delay prediction model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Whether model loaded successfully
        """
        try:
            # Try to load LightGBM model
            # In production, this would load from S3 or Triton
            import lightgbm as lgb
            
            if model_path:
                self._model = lgb.Booster(model_file=model_path)
                self._model_loaded = True
                logger.info("delay_model_loaded", path=model_path)
                return True
            
        except ImportError:
            logger.warning("lightgbm_not_available", using="heuristic_prediction")
        except Exception as e:
            logger.error("model_load_failed", error=str(e))
        
        self._model_loaded = False
        return False
    
    def _extract_features(
        self,
        distance_meters: float,
        historical_duration_seconds: float,
        context: TravelContext,
        route_confidence: float,
        departure_time: Optional[datetime] = None,
    ) -> np.ndarray:
        """
        Extract features for the ML model.
        
        Args:
            distance_meters: Route distance
            historical_duration_seconds: Historical average duration
            context: Travel context with weather/traffic
            route_confidence: Confidence in route matching
            departure_time: Planned departure time
            
        Returns:
            Feature array
        """
        now = departure_time or datetime.now(timezone.utc)
        
        # Time features
        hour = now.hour
        day_of_week = now.weekday()
        is_rush_hour = 1.0 if hour in [7, 8, 9, 17, 18, 19] else 0.0
        
        # Weather impact
        weather_impact = 0.0
        if context.weather:
            weather_impact = context.weather.travel_impact_score
        
        # Traffic delay factor
        traffic_delay = context.combined_delay_factor
        
        features = np.array([
            distance_meters,
            historical_duration_seconds,
            weather_impact,
            traffic_delay,
            float(hour),
            float(day_of_week),
            is_rush_hour,
            route_confidence,
        ], dtype=np.float32)
        
        return features.reshape(1, -1)
    
    def _heuristic_prediction(
        self,
        distance_meters: float,
        historical_duration_seconds: float,
        context: TravelContext,
    ) -> Tuple[float, float]:
        """
        Fallback heuristic prediction when model unavailable.
        
        Args:
            distance_meters: Route distance
            historical_duration_seconds: Historical duration
            context: Travel context
            
        Returns:
            Tuple of (delay_minutes, confidence)
        """
        # Base delay from traffic
        traffic_delay_factor = context.combined_delay_factor
        
        if historical_duration_seconds > 0:
            base_duration = historical_duration_seconds
        else:
            # Estimate duration from distance
            avg_speed_ms = DEFAULT_MIXED_SPEED_KMH * 1000 / 3600
            base_duration = distance_meters / avg_speed_ms
        
        # Calculate delay
        expected_duration = base_duration * traffic_delay_factor
        delay_seconds = expected_duration - base_duration
        delay_minutes = max(0, delay_seconds / 60)
        
        # Add weather impact
        if context.weather:
            weather_factor = 1.0 + context.weather.travel_impact_score * 0.3
            delay_minutes *= weather_factor
        
        # Confidence based on data availability
        confidence = context.context_confidence * 0.8
        
        return delay_minutes, confidence
    
    async def predict(
        self,
        distance_meters: float,
        historical_duration_seconds: float,
        context: TravelContext,
        route_confidence: float,
        departure_time: Optional[datetime] = None,
    ) -> Tuple[float, float, float]:
        """
        Predict travel delay.
        
        Args:
            distance_meters: Route distance
            historical_duration_seconds: Historical average
            context: Travel context
            route_confidence: Route matching confidence
            departure_time: Planned departure
            
        Returns:
            Tuple of (delay_minutes, confidence, inference_time_ms)
        """
        start_time = time.time()
        
        if self._model_loaded and self._model:
            # Use ML model
            features = self._extract_features(
                distance_meters,
                historical_duration_seconds,
                context,
                route_confidence,
                departure_time,
            )
            
            try:
                # Model predicts delay in minutes
                prediction = self._model.predict(features)[0]
                delay_minutes = max(0, float(prediction))
                confidence = 0.85 * context.context_confidence
                
            except Exception as e:
                logger.error("model_prediction_failed", error=str(e))
                delay_minutes, confidence = self._heuristic_prediction(
                    distance_meters, historical_duration_seconds, context
                )
        else:
            # Fallback to heuristic
            delay_minutes, confidence = self._heuristic_prediction(
                distance_meters, historical_duration_seconds, context
            )
        
        inference_time = (time.time() - start_time) * 1000
        
        return delay_minutes, confidence, inference_time


class PredictiveRoutingService:
    """
    Main service for predictive routing.
    
    Combines behavioral profiles, external data, and ML predictions.
    """
    
    def __init__(
        self,
        fusion_service: Optional[TravelContextFusion] = None,
    ):
        """
        Initialize predictive routing service.
        
        Args:
            fusion_service: Travel context fusion service
        """
        self.fusion = fusion_service or get_travel_context_fusion()
        self.delay_predictor = DelayPredictor()
        
        # Cache for behavioral profiles
        self._profile_cache: Dict[int, BehavioralProfile] = {}
    
    def set_profile(self, user_id: int, profile: BehavioralProfile) -> None:
        """
        Set behavioral profile for a user.
        
        Args:
            user_id: User ID
            profile: Behavioral profile
        """
        self._profile_cache[user_id] = profile
    
    def _match_known_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        profile: Optional[BehavioralProfile],
    ) -> Tuple[Optional[RoutePattern], float]:
        """
        Match origin/destination to known routes.
        
        Args:
            origin: Origin (lat, lon)
            destination: Destination (lat, lon)
            profile: User's behavioral profile
            
        Returns:
            Tuple of (matched_route, confidence)
        """
        if not profile or not profile.known_routes:
            return None, 0.0
        
        best_match = None
        best_confidence = 0.0
        
        for route in profile.known_routes:
            # Check start zone match
            start_dist = haversine_distance(
                origin[0], origin[1],
                route.start_zone[0], route.start_zone[1]
            )
            start_match = start_dist <= route.start_zone[2]
            
            # Check end zone match
            end_dist = haversine_distance(
                destination[0], destination[1],
                route.end_zone[0], route.end_zone[1]
            )
            end_match = end_dist <= route.end_zone[2]
            
            if start_match and end_match:
                # Calculate confidence based on frequency and distance
                freq_factor = min(1.0, route.frequency / 10)
                dist_factor = 1.0 - (start_dist + end_dist) / (
                    route.start_zone[2] + route.end_zone[2] + 1
                )
                
                confidence = (freq_factor * 0.6 + dist_factor * 0.4)
                
                if confidence > best_confidence:
                    best_match = route
                    best_confidence = confidence
        
        return best_match, best_confidence
    
    def _calculate_weather_delay(
        self,
        context: TravelContext,
        base_duration_minutes: float,
    ) -> float:
        """Calculate weather-related delay contribution."""
        if not context.weather:
            return 0.0
        
        impact = context.weather.travel_impact_score
        return base_duration_minutes * impact * 0.3  # 30% weight for weather
    
    def _calculate_traffic_delay(
        self,
        context: TravelContext,
        base_duration_minutes: float,
    ) -> float:
        """Calculate traffic-related delay contribution."""
        if not context.traffic:
            return 0.0
        
        delay_factor = context.traffic.delay_factor - 1.0
        return base_duration_minutes * delay_factor  # Direct delay from traffic
    
    def _generate_explanations(
        self,
        context: TravelContext,
        matched_route: Optional[RoutePattern],
        delay_minutes: float,
        weather_delay: float,
        traffic_delay: float,
    ) -> List[str]:
        """Generate human-readable explanations."""
        explanations = []
        
        # Route matching explanation
        if matched_route:
            explanations.append(
                f"Matched known route (seen {matched_route.frequency} times)"
            )
        else:
            explanations.append("Route not in historical patterns")
        
        # Delay breakdown
        if delay_minutes > 0:
            if traffic_delay > weather_delay and traffic_delay > 1:
                explanations.append(
                    f"Traffic contributing {traffic_delay:.1f} min delay"
                )
            if weather_delay > 1:
                explanations.append(
                    f"Weather contributing {weather_delay:.1f} min delay"
                )
        else:
            explanations.append("No significant delays expected")
        
        # Add context factors
        explanations.extend(context.factors)
        
        return explanations
    
    async def predict_delay(
        self,
        user_id: int,
        origin_lat: float,
        origin_lon: float,
        destination_lat: float,
        destination_lon: float,
        departure_time: Optional[datetime] = None,
        include_context: bool = True,
    ) -> DelayPrediction:
        """
        Predict travel delay for a trip.
        
        Args:
            user_id: User ID
            origin_lat: Origin latitude
            origin_lon: Origin longitude
            destination_lat: Destination latitude
            destination_lon: Destination longitude
            departure_time: Planned departure time
            include_context: Include travel context in response
            
        Returns:
            DelayPrediction with delay forecast
        """
        start_time = time.time()
        now = departure_time or datetime.now(timezone.utc)
        
        # Get travel context
        context = await self.fusion.get_context(
            origin_lat, origin_lon,
            destination_lat, destination_lon
        )
        
        # Calculate distance
        distance = haversine_distance(
            origin_lat, origin_lon,
            destination_lat, destination_lon
        )
        
        # Get user profile
        profile = self._profile_cache.get(user_id)
        
        # Match to known route
        origin = (origin_lat, origin_lon)
        destination = (destination_lat, destination_lon)
        matched_route, route_confidence = self._match_known_route(
            origin, destination, profile
        )
        
        # Get historical duration
        if matched_route:
            historical_duration = matched_route.avg_duration_seconds
            expected_distance = matched_route.avg_distance_meters
        else:
            # Estimate from distance
            avg_speed_ms = DEFAULT_MIXED_SPEED_KMH * 1000 / 3600
            historical_duration = distance / avg_speed_ms
            expected_distance = distance
        
        # Predict delay
        delay_minutes, delay_confidence, pred_time = await self.delay_predictor.predict(
            distance,
            historical_duration,
            context,
            route_confidence,
            now,
        )
        
        # Calculate delay breakdown
        base_duration_minutes = historical_duration / 60
        weather_delay = self._calculate_weather_delay(context, base_duration_minutes)
        traffic_delay = self._calculate_traffic_delay(context, base_duration_minutes)
        
        # Expected total duration
        expected_duration_minutes = base_duration_minutes + delay_minutes
        
        # Generate explanations
        explanations = self._generate_explanations(
            context,
            matched_route,
            delay_minutes,
            weather_delay,
            traffic_delay,
        )
        
        total_time = (time.time() - start_time) * 1000
        
        prediction = DelayPrediction(
            prediction_id=f"delay_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            route_id=matched_route.route_id if matched_route else None,
            origin=origin,
            destination=destination,
            expected_distance_meters=expected_distance,
            predicted_delay_minutes=delay_minutes,
            delay_confidence=delay_confidence,
            expected_duration_minutes=expected_duration_minutes,
            route_confidence=route_confidence,
            weather_delay_minutes=weather_delay,
            traffic_delay_minutes=traffic_delay,
            travel_context=context if include_context else None,
            explanation=explanations,
            predicted_at=now,
            model_version=self.delay_predictor._model_version,
            inference_time_ms=total_time,
        )
        
        logger.info(
            "delay_predicted",
            user_id=user_id,
            delay_minutes=delay_minutes,
            confidence=delay_confidence,
            time_ms=total_time,
        )
        
        return prediction


# Singleton instance
_routing_service: Optional[PredictiveRoutingService] = None


def get_predictive_routing_service() -> PredictiveRoutingService:
    """Get or create singleton routing service."""
    global _routing_service
    if _routing_service is None:
        _routing_service = PredictiveRoutingService()
    return _routing_service


async def predict_travel_delay(
    request: DelayPredictionRequest,
) -> DelayPrediction:
    """
    Convenience function to predict delay.
    
    Args:
        request: Delay prediction request
        
    Returns:
        DelayPrediction
    """
    service = get_predictive_routing_service()
    return await service.predict_delay(
        user_id=request.user_id,
        origin_lat=request.origin_lat,
        origin_lon=request.origin_lon,
        destination_lat=request.destination_lat,
        destination_lon=request.destination_lon,
        departure_time=request.departure_time,
        include_context=request.include_context,
    )
