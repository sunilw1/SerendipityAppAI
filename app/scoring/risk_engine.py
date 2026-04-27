"""
Risk Scoring Engine
====================

Core Phase 2 risk scoring layer.

Aggregates:
- Phase 1 confidence scores
- Anomaly detection signals
- Spoofing detection signals
- Baseline deviation metrics

Produces:
- Location-level risk scores (0-1)
- Session-level risk scores (0-1)
- Explainable risk factors

Design Principles:
- Continuous scores (0-1), never binary
- Confidence-calibrated
- Explainable by design
- Tunable without code changes
- Low false positive rate
"""

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.core.logging import get_logger
from app.models.schemas import (
    NormalizedTrackingEvent,
    ComputedFeatures,
    ConfidenceScore,
    UserBaseline,
)
from app.models.phase2_schemas import (
    RiskLevel,
    RiskContributor,
    LocationRiskScore,
    SessionRiskScore,
    AnomalyResult,
    SpoofingResult,
    BehavioralProfile,
    Phase2Intelligence,
)

logger = get_logger(__name__)


class RiskWeights:
    """
    Configurable weights for risk score calculation.
    
    Can be tuned without code changes.
    """
    # Location-level weights
    CONFIDENCE_INVERSE: float = 0.30  # Low confidence = higher risk
    ANOMALY_SCORE: float = 0.30       # Anomaly detection contribution
    SPOOFING_SCORE: float = 0.25      # Spoofing detection contribution
    BASELINE_DEVIATION: float = 0.15   # Deviation from learned baseline
    
    # Session-level weights
    SESSION_ANOMALY_RATE: float = 0.25
    SESSION_MAX_RISK: float = 0.20
    SESSION_AVG_RISK: float = 0.20
    SESSION_SPOOFING: float = 0.20
    SESSION_INTEGRITY: float = 0.15
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that weights sum to approximately 1.0."""
        location_sum = (
            cls.CONFIDENCE_INVERSE +
            cls.ANOMALY_SCORE +
            cls.SPOOFING_SCORE +
            cls.BASELINE_DEVIATION
        )
        session_sum = (
            cls.SESSION_ANOMALY_RATE +
            cls.SESSION_MAX_RISK +
            cls.SESSION_AVG_RISK +
            cls.SESSION_SPOOFING +
            cls.SESSION_INTEGRITY
        )
        return abs(location_sum - 1.0) < 0.01 and abs(session_sum - 1.0) < 0.01


class LocationRiskCalculator:
    """
    Calculates risk scores for individual location points.
    
    Combines Phase 1 confidence with Phase 2 detection signals.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        risk_threshold_high: float = 0.7,
        risk_threshold_moderate: float = 0.4,
    ):
        """
        Initialize location risk calculator.
        
        Args:
            weights: Custom weights (uses RiskWeights defaults if None)
            risk_threshold_high: Threshold for high risk
            risk_threshold_moderate: Threshold for moderate risk
        """
        self.weights = weights or {
            "confidence_inverse": RiskWeights.CONFIDENCE_INVERSE,
            "anomaly": RiskWeights.ANOMALY_SCORE,
            "spoofing": RiskWeights.SPOOFING_SCORE,
            "baseline": RiskWeights.BASELINE_DEVIATION,
        }
        self.high_threshold = risk_threshold_high
        self.moderate_threshold = risk_threshold_moderate
    
    def calculate(
        self,
        event: NormalizedTrackingEvent,
        confidence: ConfidenceScore,
        anomaly: Optional[AnomalyResult],
        spoofing: Optional[SpoofingResult],
        baseline: Optional[UserBaseline],
        profile: Optional[BehavioralProfile] = None,
    ) -> LocationRiskScore:
        """
        Calculate risk score for a single location.
        
        Args:
            event: Normalized event
            confidence: Phase 1 confidence score
            anomaly: Anomaly detection result
            spoofing: Spoofing detection result
            baseline: User's learned baseline
            profile: Extended behavioral profile
            
        Returns:
            LocationRiskScore with explanation
        """
        contributors = []
        reasons = []
        
        # 1. Confidence-inverse factor (low confidence = higher risk)
        confidence_factor = 1.0 - confidence.overall
        contributors.append(RiskContributor(
            factor="phase1_confidence",
            weight=self.weights["confidence_inverse"],
            score=confidence_factor,
            weighted_contribution=self.weights["confidence_inverse"] * confidence_factor,
            description=f"Phase 1 confidence: {confidence.overall:.2f}",
        ))
        
        if confidence.overall < 0.5:
            reasons.append(f"low_confidence_{confidence.confidence_level}")
        
        # 2. Anomaly factor
        if anomaly is not None:
            anomaly_factor = anomaly.anomaly_score
            contributors.append(RiskContributor(
                factor="anomaly_detection",
                weight=self.weights["anomaly"],
                score=anomaly_factor,
                weighted_contribution=self.weights["anomaly"] * anomaly_factor,
                description=f"Anomaly score: {anomaly.anomaly_score:.2f}",
            ))
            
            if anomaly.is_anomalous:
                # Add top anomaly reasons
                for signal in sorted(anomaly.signals, key=lambda s: s.severity, reverse=True)[:3]:
                    reasons.append(signal.anomaly_type.value)
        else:
            anomaly_factor = 0.0
            contributors.append(RiskContributor(
                factor="anomaly_detection",
                weight=self.weights["anomaly"],
                score=0.0,
                weighted_contribution=0.0,
                description="No anomaly detected",
            ))
        
        # 3. Spoofing factor
        if spoofing is not None:
            spoofing_factor = max(spoofing.spoofing_likelihood, spoofing.tampering_likelihood)
            contributors.append(RiskContributor(
                factor="spoofing_detection",
                weight=self.weights["spoofing"],
                score=spoofing_factor,
                weighted_contribution=self.weights["spoofing"] * spoofing_factor,
                description=f"Spoofing likelihood: {spoofing.spoofing_likelihood:.2f}",
            ))
            
            if spoofing.requires_attention:
                for indicator in spoofing.indicators[:3]:
                    reasons.append(indicator.indicator.value)
        else:
            spoofing_factor = 0.0
            contributors.append(RiskContributor(
                factor="spoofing_detection",
                weight=self.weights["spoofing"],
                score=0.0,
                weighted_contribution=0.0,
                description="No spoofing detected",
            ))
        
        # 4. Baseline deviation factor
        baseline_factor = self._calculate_baseline_deviation(
            event, baseline, profile
        )
        contributors.append(RiskContributor(
            factor="baseline_deviation",
            weight=self.weights["baseline"],
            score=baseline_factor,
            weighted_contribution=self.weights["baseline"] * baseline_factor,
            description=f"Baseline deviation: {baseline_factor:.2f}",
        ))
        
        if baseline_factor > 0.5:
            reasons.append("deviation_from_baseline")
        
        # Calculate weighted risk score
        risk_score = sum(c.weighted_contribution for c in contributors)
        risk_score = min(1.0, max(0.0, risk_score))
        
        # Determine risk level
        risk_level = self._score_to_level(risk_score)
        
        return LocationRiskScore(
            event_id=event.event_id,
            timestamp=event.timestamp,
            location=event.location,
            risk_score=risk_score,
            risk_level=risk_level,
            contributors=contributors,
            confidence_factor=confidence_factor,
            anomaly_factor=anomaly_factor,
            spoofing_factor=spoofing_factor,
            baseline_factor=baseline_factor,
            reasons=list(set(reasons)),  # Deduplicate
        )
    
    def _calculate_baseline_deviation(
        self,
        event: NormalizedTrackingEvent,
        baseline: Optional[UserBaseline],
        profile: Optional[BehavioralProfile],
    ) -> float:
        """Calculate deviation from learned baseline."""
        if baseline is None or not baseline.is_mature:
            return 0.0  # No baseline to compare
        
        deviations = []
        
        # Speed deviation
        if event.speed_ms is not None:
            speed_mean = baseline.speed_baseline.mean
            speed_std = baseline.speed_baseline.std
            
            if speed_std > 0:
                z_score = abs(event.speed_ms - speed_mean) / speed_std
                deviation = min(1.0, z_score / 3.0)  # Normalize to 0-1
                deviations.append(deviation)
        
        # Activity deviation
        activity = event.activity_type.value
        if activity in baseline.activity_distribution:
            activity_prob = baseline.activity_distribution[activity]
            # Low probability activity = higher deviation
            if activity_prob < 0.1:
                deviations.append(0.5)
            elif activity_prob < 0.05:
                deviations.append(0.7)
        
        # Time of day deviation (if profile available)
        if profile and profile.hourly_patterns:
            hour = event.timestamp.hour
            hour_pattern = next(
                (p for p in profile.hourly_patterns if p.hour == hour),
                None
            )
            if hour_pattern and hour_pattern.activity_probability < 0.05:
                deviations.append(0.3)  # Unusual time
        
        if deviations:
            return np.mean(deviations)
        return 0.0
    
    def _score_to_level(self, score: float) -> RiskLevel:
        """Convert numeric score to risk level."""
        if score < 0.1:
            return RiskLevel.MINIMAL
        elif score < 0.25:
            return RiskLevel.LOW
        elif score < 0.45:
            return RiskLevel.MODERATE
        elif score < 0.65:
            return RiskLevel.ELEVATED
        elif score < 0.85:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL


class SessionRiskCalculator:
    """
    Calculates aggregated risk scores for sessions/trips.
    
    Combines individual location risks into session-level assessment.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        high_risk_threshold: float = 0.7,
        max_flagged_events: int = 10,
    ):
        """
        Initialize session risk calculator.
        
        Args:
            weights: Custom weights
            high_risk_threshold: Threshold for high-risk points
            max_flagged_events: Max events to include in flagged list
        """
        self.weights = weights or {
            "anomaly_rate": RiskWeights.SESSION_ANOMALY_RATE,
            "max_risk": RiskWeights.SESSION_MAX_RISK,
            "avg_risk": RiskWeights.SESSION_AVG_RISK,
            "spoofing": RiskWeights.SESSION_SPOOFING,
            "integrity": RiskWeights.SESSION_INTEGRITY,
        }
        self.high_risk_threshold = high_risk_threshold
        self.max_flagged = max_flagged_events
    
    def calculate(
        self,
        session_id: str,
        user_id: int,
        location_risks: List[LocationRiskScore],
        session_spoofing: Optional[SpoofingResult],
    ) -> SessionRiskScore:
        """
        Calculate session-level risk score.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            location_risks: Risk scores for each location
            session_spoofing: Session-level spoofing result
            
        Returns:
            SessionRiskScore with explanation
        """
        if not location_risks:
            return self._empty_session_risk(session_id, user_id)
        
        contributors = []
        reasons = []
        
        # Extract location risk scores
        scores = [lr.risk_score for lr in location_risks]
        
        # 1. Anomaly rate factor
        anomalous_count = sum(1 for s in scores if s >= self.high_risk_threshold)
        anomaly_rate = anomalous_count / len(scores)
        
        anomaly_rate_factor = min(1.0, anomaly_rate * 2)  # Scale up
        contributors.append(RiskContributor(
            factor="anomaly_rate",
            weight=self.weights["anomaly_rate"],
            score=anomaly_rate_factor,
            weighted_contribution=self.weights["anomaly_rate"] * anomaly_rate_factor,
            description=f"{anomalous_count}/{len(scores)} high-risk points ({anomaly_rate*100:.1f}%)",
        ))
        
        if anomaly_rate > 0.1:
            reasons.append(f"high_anomaly_rate_{anomaly_rate*100:.0f}pct")
        
        # 2. Maximum risk factor
        max_risk = max(scores)
        contributors.append(RiskContributor(
            factor="max_location_risk",
            weight=self.weights["max_risk"],
            score=max_risk,
            weighted_contribution=self.weights["max_risk"] * max_risk,
            description=f"Maximum location risk: {max_risk:.2f}",
        ))
        
        if max_risk >= 0.8:
            reasons.append("critical_risk_point_detected")
        
        # 3. Average risk factor
        avg_risk = np.mean(scores)
        contributors.append(RiskContributor(
            factor="avg_location_risk",
            weight=self.weights["avg_risk"],
            score=avg_risk,
            weighted_contribution=self.weights["avg_risk"] * avg_risk,
            description=f"Average location risk: {avg_risk:.2f}",
        ))
        
        if avg_risk > 0.5:
            reasons.append("elevated_average_risk")
        
        # 4. Spoofing factor
        if session_spoofing is not None:
            spoofing_factor = session_spoofing.spoofing_likelihood
            integrity_factor = 1.0 - session_spoofing.session_integrity_score
            
            contributors.append(RiskContributor(
                factor="spoofing_likelihood",
                weight=self.weights["spoofing"],
                score=spoofing_factor,
                weighted_contribution=self.weights["spoofing"] * spoofing_factor,
                description=f"Session spoofing likelihood: {spoofing_factor:.2f}",
            ))
            
            contributors.append(RiskContributor(
                factor="integrity_issues",
                weight=self.weights["integrity"],
                score=integrity_factor,
                weighted_contribution=self.weights["integrity"] * integrity_factor,
                description=f"Session integrity: {session_spoofing.session_integrity_score:.2f}",
            ))
            
            if session_spoofing.requires_attention:
                reasons.append("session_spoofing_detected")
            
            if session_spoofing.session_integrity_score < 0.8:
                reasons.append("session_integrity_issues")
        else:
            spoofing_factor = 0.0
            integrity_factor = 0.0
            contributors.append(RiskContributor(
                factor="spoofing_likelihood",
                weight=self.weights["spoofing"],
                score=0.0,
                weighted_contribution=0.0,
                description="No session-level spoofing detected",
            ))
            contributors.append(RiskContributor(
                factor="integrity_issues",
                weight=self.weights["integrity"],
                score=0.0,
                weighted_contribution=0.0,
                description="Session integrity OK",
            ))
        
        # Calculate weighted session risk
        session_risk = sum(c.weighted_contribution for c in contributors)
        session_risk = min(1.0, max(0.0, session_risk))
        
        # Determine risk level
        risk_level = self._score_to_level(session_risk)
        
        # Get flagged events (highest risk)
        sorted_risks = sorted(
            location_risks,
            key=lambda lr: lr.risk_score,
            reverse=True
        )
        flagged_ids = [
            lr.event_id for lr in sorted_risks[:self.max_flagged]
            if lr.risk_score >= 0.5
        ]
        
        # Timing
        start_time = location_risks[0].timestamp
        end_time = location_risks[-1].timestamp
        
        return SessionRiskScore(
            session_id=session_id,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            risk_score=session_risk,
            risk_level=risk_level,
            total_points=len(location_risks),
            high_risk_points=anomalous_count,
            anomalous_points=sum(1 for lr in location_risks if lr.anomaly_factor > 0.5),
            max_location_risk=max_risk,
            avg_location_risk=avg_risk,
            spoofing_likelihood=spoofing_factor,
            integrity_score=1.0 - integrity_factor,
            contributors=contributors,
            reasons=list(set(reasons)),
            flagged_event_ids=flagged_ids,
        )
    
    def _empty_session_risk(
        self,
        session_id: str,
        user_id: int,
    ) -> SessionRiskScore:
        """Create empty session risk for empty sessions."""
        now = datetime.now(timezone.utc)
        return SessionRiskScore(
            session_id=session_id,
            user_id=user_id,
            start_time=now,
            end_time=now,
            risk_score=0.0,
            risk_level=RiskLevel.MINIMAL,
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
        )
    
    def _score_to_level(self, score: float) -> RiskLevel:
        """Convert numeric score to risk level."""
        if score < 0.1:
            return RiskLevel.MINIMAL
        elif score < 0.25:
            return RiskLevel.LOW
        elif score < 0.45:
            return RiskLevel.MODERATE
        elif score < 0.65:
            return RiskLevel.ELEVATED
        elif score < 0.85:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL


class RiskScoringEngine:
    """
    Main risk scoring engine for Phase 2.
    
    Orchestrates location and session risk calculation,
    integrating all detection signals.
    """
    
    def __init__(
        self,
        location_weights: Optional[Dict[str, float]] = None,
        session_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize risk scoring engine.
        
        Args:
            location_weights: Custom weights for location risk
            session_weights: Custom weights for session risk
        """
        self.location_calculator = LocationRiskCalculator(weights=location_weights)
        self.session_calculator = SessionRiskCalculator(weights=session_weights)
    
    def score_location(
        self,
        event: NormalizedTrackingEvent,
        confidence: ConfidenceScore,
        anomaly: Optional[AnomalyResult],
        spoofing: Optional[SpoofingResult],
        baseline: Optional[UserBaseline],
        profile: Optional[BehavioralProfile] = None,
    ) -> LocationRiskScore:
        """Score risk for a single location."""
        return self.location_calculator.calculate(
            event, confidence, anomaly, spoofing, baseline, profile
        )
    
    def score_locations_batch(
        self,
        events: List[NormalizedTrackingEvent],
        confidences: List[ConfidenceScore],
        anomalies: List[Optional[AnomalyResult]],
        spoofings: List[Optional[SpoofingResult]],
        baseline: Optional[UserBaseline],
        profile: Optional[BehavioralProfile] = None,
    ) -> List[LocationRiskScore]:
        """Score risk for a batch of locations."""
        return [
            self.score_location(e, c, a, s, baseline, profile)
            for e, c, a, s in zip(events, confidences, anomalies, spoofings)
        ]
    
    def score_session(
        self,
        session_id: str,
        user_id: int,
        location_risks: List[LocationRiskScore],
        session_spoofing: Optional[SpoofingResult],
    ) -> SessionRiskScore:
        """Score risk for a session."""
        return self.session_calculator.calculate(
            session_id, user_id, location_risks, session_spoofing
        )
    
    def create_intelligence(
        self,
        event: NormalizedTrackingEvent,
        session_id: str,
        confidence: ConfidenceScore,
        location_risk: LocationRiskScore,
        anomaly: Optional[AnomalyResult],
        spoofing: Optional[SpoofingResult],
    ) -> Phase2Intelligence:
        """
        Create Phase 2 intelligence output for a location.
        
        Combines all Phase 1 and Phase 2 outputs into
        a unified intelligence record.
        """
        # Determine if this point can be trusted
        is_trusted = (
            location_risk.risk_score < 0.5 and
            confidence.overall >= 0.5 and
            (anomaly is None or not anomaly.is_anomalous) and
            (spoofing is None or not spoofing.requires_attention)
        )
        
        # Get anomaly types
        anomaly_types = []
        if anomaly and anomaly.signals:
            anomaly_types = [s.anomaly_type.value for s in anomaly.signals]
        
        return Phase2Intelligence(
            event_id=event.event_id,
            user_id=event.user_id,
            session_id=session_id,
            timestamp=event.timestamp,
            location=event.location,
            phase1_confidence=confidence.overall,
            confidence_level=confidence.confidence_level,
            risk_score=location_risk.risk_score,
            risk_level=location_risk.risk_level,
            anomaly_score=anomaly.anomaly_score if anomaly else 0.0,
            spoofing_likelihood=spoofing.spoofing_likelihood if spoofing else 0.0,
            risk_reasons=location_risk.reasons,
            anomaly_types=anomaly_types,
            is_trusted=is_trusted,
        )
    
    def create_intelligence_batch(
        self,
        events: List[NormalizedTrackingEvent],
        session_id: str,
        confidences: List[ConfidenceScore],
        location_risks: List[LocationRiskScore],
        anomalies: List[Optional[AnomalyResult]],
        spoofings: List[Optional[SpoofingResult]],
    ) -> List[Phase2Intelligence]:
        """Create intelligence outputs for a batch."""
        return [
            self.create_intelligence(e, session_id, c, lr, a, s)
            for e, c, lr, a, s in zip(
                events, confidences, location_risks, anomalies, spoofings
            )
        ]
