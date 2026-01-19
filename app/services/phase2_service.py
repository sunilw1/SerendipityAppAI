"""
Phase 2 Intelligence Service
=============================

Main service for Phase 2 behavioral intelligence and threat detection.

Orchestrates:
1. Behavioral profile learning
2. Anomaly detection
3. Spoofing detection
4. Risk scoring
5. Explainability generation

Consumes Phase 1 outputs:
- Baselines
- Confidence scores
- Processed events

Produces Phase 2 intelligence:
- Risk scores (location + session)
- Anomaly flags with explanations
- Spoofing likelihood
- Validated movement signals

Design Principles:
- Backend-only processing
- No mobile/on-device computation
- Battery-safe (runs on server)
- Scores + explanations, not hard decisions
- Explainability is mandatory
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
import threading

import numpy as np

from app.core.logging import get_logger, log_processing_metrics
from app.models.schemas import (
    NormalizedTrackingEvent,
    ComputedFeatures,
    ConfidenceScore,
    UserBaseline,
    ProcessedTrackingEvent,
    Trip,
)
from app.models.phase2_schemas import (
    BehavioralProfile,
    AnomalyResult,
    SpoofingResult,
    LocationRiskScore,
    SessionRiskScore,
    Phase2Intelligence,
    SessionIntelligence,
    RiskAnalysisResponse,
    AnomalyDetectionResponse,
    SpoofingDetectionResponse,
)
from app.models.behavioral_profile import BehavioralProfileLearner
from app.detection.anomaly import EnsembleAnomalyDetector
from app.detection.spoofing import SpoofingDetector, TamperingDetector
from app.scoring.risk_engine import RiskScoringEngine
from app.scoring.explainability import ExplainabilityEngine
from app.inference.accelerator import InferenceAccelerator
from app.services.baseline_service import BaselineService, get_baseline_service
from app.services.intelligence_service import IntelligenceService, get_intelligence_service

logger = get_logger(__name__)


class Phase2Service:
    """
    Core Phase 2 intelligence service.
    
    Provides end-to-end behavioral intelligence:
    Phase 1 Outputs → Analysis → Risk Scores → Explanations
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        anomaly_threshold: float = 0.5,
        risk_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize Phase 2 service.
        
        Args:
            use_gpu: Enable GPU acceleration
            anomaly_threshold: Threshold for anomaly detection
            risk_weights: Custom risk weights
        """
        # Phase 1 services
        self.baseline_service = get_baseline_service()
        self.intelligence_service = get_intelligence_service()
        
        # Phase 2 components
        self.profile_learner = BehavioralProfileLearner()
        self.anomaly_detector = EnsembleAnomalyDetector(
            anomaly_threshold=anomaly_threshold
        )
        self.spoofing_detector = SpoofingDetector()
        self.tampering_detector = TamperingDetector()
        self.risk_engine = RiskScoringEngine(
            location_weights=risk_weights
        )
        self.explainability = ExplainabilityEngine()
        
        # GPU acceleration
        self.accelerator = InferenceAccelerator(prefer_gpu=use_gpu)
        
        # Caches
        self._profiles: Dict[int, BehavioralProfile] = {}
        self._session_cache: Dict[str, SessionIntelligence] = {}
        
        # Lock to prevent concurrent processing of the same user
        self._processing_locks: Dict[int, threading.Lock] = {}
        self._locks_lock = threading.Lock()  # Lock for creating user locks
        
        logger.info("phase2_service_initialized", use_gpu=use_gpu)
    
    def _get_user_lock(self, user_id: int) -> threading.Lock:
        """Get or create a lock for a specific user."""
        with self._locks_lock:
            if user_id not in self._processing_locks:
                self._processing_locks[user_id] = threading.Lock()
            return self._processing_locks[user_id]
    
    def process_user(
        self,
        user_id: int,
        force_relearn: bool = False,
    ) -> Tuple[BehavioralProfile, List[SessionIntelligence]]:
        """
        Process a user's data through Phase 2 pipeline.
        
        Args:
            user_id: User identifier
            force_relearn: Force relearning of profile
            
        Returns:
            Tuple of (behavioral profile, session intelligence list)
        """
        import time
        
        # Quick cache check before acquiring lock
        if not force_relearn and user_id in self._profiles:
            cached_sessions = [
                self._session_cache[sid]
                for sid in self._session_cache
                if self._session_cache[sid].user_id == user_id
            ]
            if cached_sessions:
                logger.info("phase2_using_cache", user_id=user_id, sessions=len(cached_sessions))
                return self._profiles[user_id], cached_sessions
        
        # Acquire lock for this user - prevents concurrent processing
        user_lock = self._get_user_lock(user_id)
        with user_lock:
            # Check cache again inside lock (another thread may have finished)
            if not force_relearn and user_id in self._profiles:
                cached_sessions = [
                    self._session_cache[sid]
                    for sid in self._session_cache
                    if self._session_cache[sid].user_id == user_id
                ]
                if cached_sessions:
                    logger.info("phase2_using_cache_after_lock", user_id=user_id, sessions=len(cached_sessions))
                    return self._profiles[user_id], cached_sessions
            
            start_time = time.time()
            logger.info("phase2_processing_user", user_id=user_id)
            
            # Get Phase 1 outputs (expensive!)
            trips, baseline = self.intelligence_service.process_user_data(user_id)
            
            if not trips:
                raise ValueError(f"No trips found for user {user_id}")
            
            # Prepare data for Phase 2
            trip_data = self._prepare_trip_data(trips)
            
            # Learn or retrieve behavioral profile
            if force_relearn or user_id not in self._profiles:
                profile = self._learn_profile(user_id, baseline, trip_data)
            else:
                profile = self._profiles[user_id]
            
            # Fit anomaly detector
            all_events = []
            all_features = []
            all_scores = []
            
            for trip_events, trip_features, trip_scores in zip(
                trip_data["events"],
                trip_data["features"],
                trip_data["scores"],
            ):
                all_events.extend(trip_events)
                all_features.extend(trip_features)
                all_scores.extend(trip_scores)
            
            self.anomaly_detector.fit(
                all_events, all_features, all_scores, baseline
            )
            
            # Process each session IN PARALLEL for speed
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            total_trips = len(trips)
            logger.info("phase2_starting_session_processing", user_id=user_id, total_sessions=total_trips)
            
            # Prepare session args
            session_args = []
            for i, trip in enumerate(trips):
                session_args.append({
                    "session_id": f"u{user_id}_trip_{trip.summary.trip_id}",
                    "user_id": user_id,
                    "events": trip_data["events"][i],
                    "features": trip_data["features"][i],
                    "scores": trip_data["scores"][i],
                    "baseline": baseline,
                    "profile": profile,
                })
            
            # Use 4 parallel workers to speed up processing
            session_intelligence = [None] * total_trips
            completed = 0
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all tasks
                future_to_idx = {
                    executor.submit(
                        self._process_session,
                        args["session_id"],
                        args["user_id"],
                        args["events"],
                        args["features"],
                        args["scores"],
                        args["baseline"],
                        args["profile"],
                    ): i
                    for i, args in enumerate(session_args)
                }
                
                # Collect results
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    session_intelligence[idx] = future.result()
                    completed += 1
                    
                    # Progress logging every 20 sessions
                    if completed % 20 == 0:
                        logger.info("phase2_session_progress", user_id=user_id, processed=completed, total=total_trips)
            
            # Cache all session results
            for session_intel in session_intelligence:
                self._session_cache[session_intel.session_id] = session_intel
            
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                "phase2_processing_complete",
                user_id=user_id,
                sessions=len(session_intelligence),
                duration_seconds=round(duration_ms / 1000, 1),
            )
            
            log_processing_metrics(
                logger,
                "phase2_process_user",
                len(all_events),
                duration_ms,
                extra={"user_id": user_id, "sessions": len(session_intelligence)}
            )
            
            return profile, session_intelligence
    
    def analyze_session(
        self,
        user_id: int,
        session_id: str,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
        scores: List[ConfidenceScore],
    ) -> SessionIntelligence:
        """
        Analyze a single session.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            events: Normalized events
            features: Computed features
            scores: Confidence scores
            
        Returns:
            SessionIntelligence
        """
        baseline = self.baseline_service.get_baseline(user_id)
        profile = self._profiles.get(user_id)
        
        return self._process_session(
            session_id=session_id,
            user_id=user_id,
            events=events,
            features=features,
            scores=scores,
            baseline=baseline,
            profile=profile,
        )
    
    def get_risk_analysis(
        self,
        user_id: int,
        session_id: Optional[str] = None,
        include_locations: bool = True,
    ) -> RiskAnalysisResponse:
        """
        Get risk analysis for a user.
        
        Args:
            user_id: User identifier
            session_id: Optional specific session
            include_locations: Include location-level risks
            
        Returns:
            RiskAnalysisResponse
        """
        # Process user if not cached
        if user_id not in self._profiles:
            profile, sessions = self.process_user(user_id)
        else:
            # Get cached sessions
            sessions = [
                self._session_cache[sid]
                for sid in self._session_cache
                if self._session_cache[sid].user_id == user_id
            ]
        
        # Filter by session if specified
        if session_id:
            sessions = [s for s in sessions if s.session_id == session_id]
        
        # Extract risk scores
        session_risks = [s.risk_score for s in sessions]
        
        # Location risks (if requested)
        location_risks = []
        if include_locations:
            for session in sessions:
                # Would need to cache location risks or recompute
                pass
        
        # Summary statistics
        all_risk_scores = [r.risk_score for r in session_risks]
        high_risk_count = sum(1 for r in session_risks if r.risk_score >= 0.7)
        
        return RiskAnalysisResponse(
            success=True,
            user_id=user_id,
            analysis_timestamp=datetime.now(timezone.utc),
            session_risks=session_risks,
            location_risks=location_risks,
            total_sessions=len(session_risks),
            high_risk_sessions=high_risk_count,
            avg_risk_score=np.mean(all_risk_scores) if all_risk_scores else 0.0,
        )
    
    def get_anomaly_detection(
        self,
        user_id: int,
        session_id: Optional[str] = None,
        sensitivity: float = 0.5,
    ) -> AnomalyDetectionResponse:
        """
        Get anomaly detection results.
        
        Args:
            user_id: User identifier
            session_id: Optional specific session
            sensitivity: Detection sensitivity
            
        Returns:
            AnomalyDetectionResponse
        """
        # Ensure user is processed
        if user_id not in self._profiles:
            self.process_user(user_id)
        
        # Get processed data
        trips, baseline = self.intelligence_service.process_user_data(user_id)
        trip_data = self._prepare_trip_data(trips)
        
        # Detect anomalies
        all_anomalies = []
        all_events = []
        
        for trip_events, trip_features, trip_scores in zip(
            trip_data["events"],
            trip_data["features"],
            trip_data["scores"],
        ):
            anomalies = self.anomaly_detector.detect_sequence_anomalies(
                trip_events, trip_features, trip_scores
            )
            
            all_anomalies.extend(anomalies)
            all_events.extend(trip_events)
        
        # Filter to anomalous only
        anomalous = [a for a in all_anomalies if a.is_anomalous]
        
        # Count by type
        type_counts: Dict[str, int] = {}
        for a in anomalous:
            for signal in a.signals:
                atype = signal.anomaly_type.value
                type_counts[atype] = type_counts.get(atype, 0) + 1
        
        return AnomalyDetectionResponse(
            success=True,
            user_id=user_id,
            anomalies=anomalous,
            total_points_analyzed=len(all_events),
            anomalous_points=len(anomalous),
            anomaly_rate=len(anomalous) / len(all_events) if all_events else 0.0,
            anomaly_type_breakdown=type_counts,
        )
    
    def get_spoofing_detection(
        self,
        user_id: int,
        session_id: Optional[str] = None,
    ) -> SpoofingDetectionResponse:
        """
        Get spoofing detection results.
        
        Args:
            user_id: User identifier
            session_id: Optional specific session
            
        Returns:
            SpoofingDetectionResponse
        """
        # Ensure user is processed
        trips, baseline = self.intelligence_service.process_user_data(user_id)
        trip_data = self._prepare_trip_data(trips)
        
        # Detect spoofing per session
        results = []
        max_spoofing = 0.0
        max_tampering = 0.0
        
        for i, (trip_events, trip_features) in enumerate(zip(
            trip_data["events"],
            trip_data["features"],
        )):
            session_id = f"trip_{trips[i].summary.trip_id}"
            
            spoofing_result = self.spoofing_detector.detect_session(
                trip_events, trip_features, session_id
            )
            
            results.append(spoofing_result)
            max_spoofing = max(max_spoofing, spoofing_result.spoofing_likelihood)
            max_tampering = max(max_tampering, spoofing_result.tampering_likelihood)
        
        return SpoofingDetectionResponse(
            success=True,
            user_id=user_id,
            session_id=session_id,
            overall_spoofing_likelihood=max_spoofing,
            overall_tampering_likelihood=max_tampering,
            results=results,
            requires_review=max_spoofing >= 0.5 or max_tampering >= 0.5,
            confidence_in_assessment=0.8,  # Based on model confidence
        )
    
    def get_profile(self, user_id: int) -> Optional[BehavioralProfile]:
        """Get cached behavioral profile."""
        return self._profiles.get(user_id)
    
    def get_profile_summary(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get profile summary for API."""
        profile = self._profiles.get(user_id)
        if not profile:
            return None
        
        return {
            "user_id": profile.user_id,
            "profile_version": profile.profile_version,
            "created_at": profile.created_at.isoformat(),
            "updated_at": profile.updated_at.isoformat(),
            "is_mature": profile.is_mature,
            "profile_confidence": profile.profile_confidence,
            "total_trips_analyzed": profile.total_trips_analyzed,
            "total_points_analyzed": profile.total_points_analyzed,
            "known_routes_count": len(profile.known_routes),
            "frequent_stops_count": len(profile.frequent_stops),
            "speed_consistency": profile.speed_consistency,
            "update_regularity": profile.update_regularity,
            "route_adherence": profile.route_adherence_score,
        }
    
    def _prepare_trip_data(
        self,
        trips: List[Trip],
    ) -> Dict[str, List]:
        """Extract data from trips for Phase 2 processing."""
        events_list = []
        features_list = []
        scores_list = []
        
        for trip in trips:
            trip_events = [pe.event for pe in trip.events]
            trip_features = [pe.features for pe in trip.events]
            trip_scores = [pe.confidence for pe in trip.events]
            
            events_list.append(trip_events)
            features_list.append(trip_features)
            scores_list.append(trip_scores)
        
        return {
            "events": events_list,
            "features": features_list,
            "scores": scores_list,
        }
    
    def _learn_profile(
        self,
        user_id: int,
        baseline: UserBaseline,
        trip_data: Dict[str, List],
    ) -> BehavioralProfile:
        """Learn behavioral profile from data."""
        profile = self.profile_learner.learn_profile(
            user_id=user_id,
            baseline=baseline,
            trips=trip_data["events"],
            features=trip_data["features"],
            scores=trip_data["scores"],
        )
        
        self._profiles[user_id] = profile
        
        logger.info(
            "profile_learned",
            user_id=user_id,
            is_mature=profile.is_mature,
            routes=len(profile.known_routes),
            stops=len(profile.frequent_stops),
        )
        
        return profile
    
    def _process_session(
        self,
        session_id: str,
        user_id: int,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
        scores: List[ConfidenceScore],
        baseline: Optional[UserBaseline],
        profile: Optional[BehavioralProfile],
    ) -> SessionIntelligence:
        """Process a single session through Phase 2 pipeline."""
        import time
        session_start = time.time()
        
        if not events:
            return self._empty_session_intelligence(session_id, user_id)
        
        # 1. Anomaly detection
        anomaly_results = self.anomaly_detector.detect_sequence_anomalies(
            events, features, scores
        )
        
        # 2. Spoofing detection
        spoofing_result = self.spoofing_detector.detect_session(
            events, features, session_id
        )
        
        # 3. Location-level risk scoring (optimized - uses session spoofing for all points)
        # This avoids 84K+ individual detect_point calls which is the main bottleneck
        location_risks = self.risk_engine.score_locations_batch(
            events=events,
            confidences=scores,
            anomalies=anomaly_results,
            spoofings=[spoofing_result] * len(events),  # Use session-level spoofing
            baseline=baseline,
            profile=profile,
        )
        
        # 4. Session-level risk scoring
        session_risk = self.risk_engine.score_session(
            session_id=session_id,
            user_id=user_id,
            location_risks=location_risks,
            session_spoofing=spoofing_result,
        )
        
        # 5. Create Phase 2 intelligence outputs
        intelligence_points = self.risk_engine.create_intelligence_batch(
            events=events,
            session_id=session_id,
            confidences=scores,
            location_risks=location_risks,
            anomalies=anomaly_results,
            spoofings=[spoofing_result] * len(events),  # Session-level for all
        )
        
        # Count trusted points
        trusted_count = sum(1 for p in intelligence_points if p.is_trusted)
        
        # Count anomaly types
        anomaly_types: Dict[str, int] = {}
        for a in anomaly_results:
            if a.is_anomalous:
                for signal in a.signals:
                    atype = signal.anomaly_type.value
                    anomaly_types[atype] = anomaly_types.get(atype, 0) + 1
        
        # Extract spoofing indicators
        spoofing_indicators = [
            ind.indicator.value for ind in spoofing_result.indicators
        ]
        
        # Determine session validity
        is_valid = (
            session_risk.risk_score < 0.7 and
            spoofing_result.spoofing_likelihood < 0.5 and
            trusted_count / len(events) >= 0.5
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            session_risk, spoofing_result, anomaly_results
        )
        
        session_intel = SessionIntelligence(
            session_id=session_id,
            user_id=user_id,
            start_time=events[0].timestamp,
            end_time=events[-1].timestamp,
            duration_seconds=events[-1].timestamp_unix - events[0].timestamp_unix,
            risk_score=session_risk,
            total_points=len(events),
            trusted_points=trusted_count,
            trust_rate=trusted_count / len(events),
            anomaly_count=sum(1 for a in anomaly_results if a.is_anomalous),
            anomaly_types=anomaly_types,
            spoofing_likelihood=spoofing_result.spoofing_likelihood,
            spoofing_indicators=spoofing_indicators,
            is_session_valid=is_valid,
            validation_confidence=0.8,  # Based on model confidences
            recommendations=recommendations,
        )
        
        # Cache
        self._session_cache[session_id] = session_intel
        
        session_duration_ms = (time.time() - session_start) * 1000
        if session_duration_ms > 100:  # Log slow sessions
            logger.debug("session_processed_slow", session_id=session_id, events=len(events), duration_ms=round(session_duration_ms, 1))
        
        return session_intel
    
    def _empty_session_intelligence(
        self,
        session_id: str,
        user_id: int,
    ) -> SessionIntelligence:
        """Create empty session intelligence."""
        now = datetime.now(timezone.utc)
        
        return SessionIntelligence(
            session_id=session_id,
            user_id=user_id,
            start_time=now,
            end_time=now,
            duration_seconds=0.0,
            risk_score=SessionRiskScore(
                session_id=session_id,
                user_id=user_id,
                start_time=now,
                end_time=now,
                risk_score=0.0,
                risk_level="minimal",
                total_points=0,
                high_risk_points=0,
                anomalous_points=0,
                max_location_risk=0.0,
                avg_location_risk=0.0,
                spoofing_likelihood=0.0,
                integrity_score=1.0,
                contributors=[],
                reasons=["empty_session"],
                flagged_event_ids=[],
            ),
            total_points=0,
            trusted_points=0,
            trust_rate=0.0,
            anomaly_count=0,
            anomaly_types={},
            spoofing_likelihood=0.0,
            spoofing_indicators=[],
            is_session_valid=False,
            validation_confidence=0.0,
            recommendations=["Insufficient data for analysis"],
        )
    
    def _generate_recommendations(
        self,
        session_risk: SessionRiskScore,
        spoofing: SpoofingResult,
        anomalies: List[AnomalyResult],
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if session_risk.risk_score >= 0.7:
            recommendations.append(
                "High session risk detected - review flagged events"
            )
        
        if spoofing.spoofing_likelihood >= 0.5:
            recommendations.append(
                "Possible GPS spoofing - verify with additional sources"
            )
        
        if spoofing.tampering_likelihood >= 0.5:
            recommendations.append(
                "Data tampering indicators present - check data integrity"
            )
        
        anomalous_count = sum(1 for a in anomalies if a.is_anomalous)
        anomaly_rate = anomalous_count / len(anomalies) if anomalies else 0
        
        if anomaly_rate > 0.2:
            recommendations.append(
                f"High anomaly rate ({anomaly_rate*100:.0f}%) - investigate patterns"
            )
        
        if not recommendations:
            recommendations.append("Session appears normal - no action required")
        
        return recommendations
    
    def get_visualization_data_fast(
        self,
        user_id: int,
        limit: int = 0,  # 0 means no limit
        min_confidence: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Get visualization data FAST - directly from Phase 1 outputs.
        
        This method skips the expensive Phase 2 session processing and
        returns data suitable for the timeline/metrics visualization
        using only Phase 1 confidence scores as a proxy for risk.
        
        Args:
            user_id: User identifier
            limit: Maximum points to return
            min_confidence: Minimum confidence filter
            
        Returns:
            Dictionary with visualization data
        """
        logger.info("phase2_viz_fast_start", user_id=user_id)
        
        # Get Phase 1 data (this is fast - ~7 seconds)
        trips, baseline = self.intelligence_service.process_user_data(user_id)
        
        if not trips:
            return {
                "success": True,
                "user_id": user_id,
                "total_points": 0,
                "points": [],
                "stats": {},
            }
        
        # Build points directly from Phase 1 outputs
        points = []
        total_confidence = 0.0
        confidence_counts = {"excellent": 0, "good": 0, "moderate": 0, "low": 0, "unreliable": 0}
        risk_counts = {"minimal": 0, "low": 0, "moderate": 0, "elevated": 0, "high": 0, "critical": 0}
        anomaly_proxies = 0  # Use low confidence as anomaly proxy
        spoofing_proxies = 0  # Use very low confidence as spoofing proxy
        
        for trip in trips:
            session_id = f"trip_{trip.summary.trip_id}"
            
            for pe in trip.events:
                conf = pe.confidence.overall
                
                # Apply filter
                if conf < min_confidence:
                    continue
                
                # Use inverse confidence as risk proxy (no heavy Phase 2 processing)
                risk_proxy = max(0.0, min(1.0, 1.0 - conf))
                
                # Determine risk level from proxy
                if risk_proxy < 0.1:
                    risk_level = "minimal"
                    risk_counts["minimal"] += 1
                elif risk_proxy < 0.25:
                    risk_level = "low"
                    risk_counts["low"] += 1
                elif risk_proxy < 0.45:
                    risk_level = "moderate"
                    risk_counts["moderate"] += 1
                elif risk_proxy < 0.65:
                    risk_level = "elevated"
                    risk_counts["elevated"] += 1
                elif risk_proxy < 0.85:
                    risk_level = "high"
                    risk_counts["high"] += 1
                else:
                    risk_level = "critical"
                    risk_counts["critical"] += 1
                
                # Confidence buckets
                if conf >= 0.9:
                    confidence_counts["excellent"] += 1
                elif conf >= 0.75:
                    confidence_counts["good"] += 1
                elif conf >= 0.5:
                    confidence_counts["moderate"] += 1
                elif conf >= 0.25:
                    confidence_counts["low"] += 1
                else:
                    confidence_counts["unreliable"] += 1
                
                # Anomaly/spoofing proxies
                if conf < 0.5:
                    anomaly_proxies += 1
                if conf < 0.25:
                    spoofing_proxies += 1
                
                total_confidence += conf
                
                # Determine trust
                is_trusted = conf >= 0.5 and len(pe.event.quality_flags) == 0
                
                # Generate simple explanation
                if conf >= 0.9:
                    explanation = "High confidence - normal movement"
                elif conf >= 0.7:
                    explanation = "Good confidence - appears normal"
                elif conf >= 0.5:
                    explanation = "Moderate confidence - minor uncertainties"
                elif conf >= 0.3:
                    explanation = "Low confidence - requires attention"
                else:
                    explanation = "Very low confidence - potential anomaly or spoofing"
                
                points.append({
                    "timestamp": pe.event.timestamp.isoformat(),
                    "lat": pe.event.location.lat,
                    "lon": pe.event.location.lon,
                    "confidence": conf,
                    "risk_score": risk_proxy,
                    "risk_level": risk_level,
                    "anomalies": ["low_confidence"] if conf < 0.5 else [],
                    "spoofing": conf < 0.25,
                    "spoofing_likelihood": max(0.0, 0.5 - conf) * 2 if conf < 0.5 else 0.0,
                    "is_trusted": is_trusted,
                    "explanation": explanation,
                    "session_id": session_id,
                })
                
                # Limit check (only if limit > 0)
                if limit > 0 and len(points) >= limit:
                    break
            
            if limit > 0 and len(points) >= limit:
                break
        
        # Sort by timestamp
        points.sort(key=lambda p: p["timestamp"])
        
        avg_confidence = total_confidence / len(points) if points else 0.0
        avg_risk = 1.0 - avg_confidence
        
        logger.info(
            "phase2_viz_fast_complete",
            user_id=user_id,
            points=len(points),
            avg_confidence=round(avg_confidence, 3),
        )
        
        return {
            "success": True,
            "user_id": user_id,
            "total_points": len(points),
            "points": points,
            "stats": {
                "avg_confidence": avg_confidence,
                "avg_risk": avg_risk,
                "confidence_distribution": confidence_counts,
                "risk_distribution": risk_counts,
                "anomaly_count": anomaly_proxies,
                "spoofing_count": spoofing_proxies,
                "total_sessions": len(trips),
            },
            "time_range": {
                "start": points[0]["timestamp"] if points else None,
                "end": points[-1]["timestamp"] if points else None,
            },
            "mode": "fast",
        }

    def has_cached_phase2_data(self, user_id: int) -> bool:
        """Check if full Phase 2 data exists in cache for a user."""
        if user_id not in self._profiles:
            return False
        
        # Check if we have cached sessions for this user
        user_sessions = [
            s for s in self._session_cache.values()
            if s.user_id == user_id
        ]
        return len(user_sessions) > 0
    
    def get_visualization_data_full(
        self,
        user_id: int,
        limit: int = 0,  # 0 means no limit
        min_confidence: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Get visualization data enriched with FULL Phase 2 session intelligence.
        
        This combines Phase 1 point data with Phase 2 session-level ML results
        (anomaly detection, spoofing detection, risk scoring).
        
        Args:
            user_id: User identifier
            limit: Maximum points to return (0 = no limit)
            min_confidence: Minimum confidence filter
            
        Returns:
            Dictionary with visualization data enriched with Phase 2 intelligence
        """
        logger.info("phase2_viz_full_start", user_id=user_id)
        
        # Get cached sessions (Phase 2 results)
        sessions = [
            s for s in self._session_cache.values()
            if s.user_id == user_id
        ]
        
        if not sessions:
            logger.warning("phase2_viz_full_no_sessions", user_id=user_id)
            return None  # Signal to fall back to fast mode
        
        # Build session lookup for enrichment
        session_lookup: Dict[str, Any] = {}
        for s in sessions:
            session_lookup[s.session_id] = {
                "risk_score": s.risk_score.risk_score if hasattr(s.risk_score, 'risk_score') else 0.5,
                "risk_level": s.risk_score.risk_level.value if hasattr(s.risk_score, 'risk_level') and hasattr(s.risk_score.risk_level, 'value') else "moderate",
                "anomaly_count": s.anomaly_count,
                "anomaly_types": dict(s.anomaly_types),
                "spoofing_likelihood": s.spoofing_likelihood,
                "spoofing_indicators": list(s.spoofing_indicators),
                "is_valid": s.is_session_valid,
                "total_points": s.total_points,
                "trusted_points": s.trusted_points,
            }
        
        # Get Phase 1 data for coordinates and timestamps
        trips, baseline = self.intelligence_service.process_user_data(user_id)
        
        if not trips:
            return None
        
        # Build points with Phase 2 enrichment
        points = []
        total_confidence = 0.0
        total_risk = 0.0
        confidence_counts = {"excellent": 0, "good": 0, "moderate": 0, "low": 0, "unreliable": 0}
        risk_counts = {"minimal": 0, "low": 0, "moderate": 0, "elevated": 0, "high": 0, "critical": 0}
        anomaly_types: Dict[str, int] = {}
        spoofing_count = 0
        
        for trip in trips:
            trip_session_id = f"u{user_id}_trip_{trip.summary.trip_id}"
            session_data = session_lookup.get(trip_session_id, {})
            
            # Use session-level risk if available, otherwise confidence-based
            session_risk = session_data.get("risk_score", None)
            session_spoofing = session_data.get("spoofing_likelihood", 0.0)
            session_anomalies = list(session_data.get("anomaly_types", {}).keys())
            is_spoofed = session_spoofing >= 0.5
            
            for pe in trip.events:
                conf = pe.confidence.overall
                
                if conf < min_confidence:
                    continue
                
                # Use session-level risk or confidence-based proxy
                if session_risk is not None:
                    # Blend session risk with confidence-based adjustment
                    risk = session_risk * (1.0 + (1.0 - conf) * 0.2)  # Adjust by confidence
                    risk = min(1.0, max(0.0, risk))
                else:
                    risk = max(0.0, min(1.0, 1.0 - conf))
                
                total_confidence += conf
                total_risk += risk
                
                # Risk level
                if risk < 0.1:
                    risk_level = "minimal"
                    risk_counts["minimal"] += 1
                elif risk < 0.25:
                    risk_level = "low"
                    risk_counts["low"] += 1
                elif risk < 0.45:
                    risk_level = "moderate"
                    risk_counts["moderate"] += 1
                elif risk < 0.65:
                    risk_level = "elevated"
                    risk_counts["elevated"] += 1
                elif risk < 0.85:
                    risk_level = "high"
                    risk_counts["high"] += 1
                else:
                    risk_level = "critical"
                    risk_counts["critical"] += 1
                
                # Confidence buckets
                if conf >= 0.9:
                    confidence_counts["excellent"] += 1
                elif conf >= 0.75:
                    confidence_counts["good"] += 1
                elif conf >= 0.5:
                    confidence_counts["moderate"] += 1
                elif conf >= 0.25:
                    confidence_counts["low"] += 1
                else:
                    confidence_counts["unreliable"] += 1
                
                # Count anomalies from session
                for atype in session_anomalies:
                    if atype not in anomaly_types:
                        anomaly_types[atype] = 0
                
                if is_spoofed:
                    spoofing_count += 1
                
                is_trusted = conf >= 0.5 and risk < 0.5 and not is_spoofed
                
                # Generate explanation based on Phase 2 data
                if session_anomalies:
                    explanation = f"ML detected: {', '.join(session_anomalies[:2])}"
                elif is_spoofed:
                    explanation = f"Spoofing likelihood: {session_spoofing:.0%}"
                elif risk >= 0.5:
                    explanation = "Elevated session risk from ML analysis"
                elif conf >= 0.9:
                    explanation = "High confidence - normal movement"
                else:
                    explanation = "Session analyzed - appears normal"
                
                points.append({
                    "timestamp": pe.event.timestamp.isoformat(),
                    "lat": pe.event.location.lat,
                    "lon": pe.event.location.lon,
                    "session_id": trip_session_id,
                    "confidence": round(conf, 4),
                    "risk_score": round(risk, 4),
                    "risk_level": risk_level,
                    "anomalies": session_anomalies if conf < 0.7 else [],
                    "spoofing": is_spoofed,
                    "spoofing_likelihood": round(session_spoofing, 4),
                    "is_trusted": is_trusted,
                    "explanation": explanation,
                })
        
        # Sort by timestamp
        points.sort(key=lambda p: p["timestamp"])
        
        # Apply limit only if specified
        if limit > 0:
            points = points[:limit]
        
        n = len(points)
        avg_confidence = total_confidence / n if n > 0 else 0.0
        avg_risk = total_risk / n if n > 0 else 0.0
        
        # Aggregate anomaly counts from sessions
        for s in sessions:
            for atype, count in s.anomaly_types.items():
                anomaly_types[atype] = anomaly_types.get(atype, 0) + count
        
        # Build stats
        stats = {
            "total_sessions": len(sessions),
            "avg_confidence": round(avg_confidence, 4),
            "avg_risk": round(avg_risk, 4),
            "confidence_distribution": confidence_counts,
            "risk_distribution": risk_counts,
            "anomaly_count": sum(anomaly_types.values()),
            "anomaly_types": anomaly_types,
            "spoofing_count": sum(1 for s in sessions if s.spoofing_likelihood >= 0.5),
            "trusted_points": sum(1 for p in points if p["is_trusted"]),
            "mode": "full_phase2",
        }
        
        # Time range
        if points:
            time_range = {
                "start": points[0]["timestamp"],
                "end": points[-1]["timestamp"],
            }
        else:
            time_range = {}
        
        logger.info(
            "phase2_viz_full_complete",
            user_id=user_id,
            points=n,
            sessions=len(sessions),
            avg_risk=round(avg_risk, 3),
        )
        
        return {
            "success": True,
            "user_id": user_id,
            "total_points": n,
            "points": points,
            "stats": stats,
            "time_range": time_range,
            "mode": "full_phase2",
        }


# Global service instance
_phase2_service: Optional[Phase2Service] = None


def get_phase2_service() -> Phase2Service:
    """Get or create the global Phase 2 service."""
    global _phase2_service
    if _phase2_service is None:
        _phase2_service = Phase2Service()
    return _phase2_service
