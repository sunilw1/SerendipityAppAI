"""
Explainability Layer
=====================

Generates human-readable explanations for all Phase 2 scores.

MANDATORY: Every anomaly, risk, or detection score must include:
- Why it was triggered
- Which signals contributed
- Relative weights (high-level)

Design Principles:
- Explanations are always included (not optional)
- Language is clear and non-technical where possible
- Weights and factors are transparent
- Supports debugging and audit
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime

from app.core.logging import get_logger
from app.models.phase2_schemas import (
    AnomalyType,
    SpoofingIndicator,
    RiskLevel,
    AnomalyResult,
    SpoofingResult,
    LocationRiskScore,
    SessionRiskScore,
    RiskContributor,
)

logger = get_logger(__name__)


class ExplanationTemplates:
    """Templates for generating human-readable explanations."""
    
    # Risk level explanations
    RISK_LEVELS = {
        RiskLevel.MINIMAL: "This location has minimal risk indicators.",
        RiskLevel.LOW: "This location has low risk with minor concerns.",
        RiskLevel.MODERATE: "This location has moderate risk and warrants review.",
        RiskLevel.ELEVATED: "This location has elevated risk with multiple concerns.",
        RiskLevel.HIGH: "This location has high risk requiring attention.",
        RiskLevel.CRITICAL: "This location has critical risk indicators.",
    }
    
    # Anomaly type explanations
    ANOMALY_TYPES = {
        AnomalyType.ROUTE_DEVIATION: "Movement deviates from typical routes",
        AnomalyType.SPEED_ANOMALY: "Speed is unusual compared to baseline",
        AnomalyType.IMPOSSIBLE_JUMP: "Location jump is physically impossible",
        AnomalyType.STOP_PATTERN_ANOMALY: "Stop pattern differs from normal behavior",
        AnomalyType.TIME_OF_DAY_ANOMALY: "Activity at unusual time of day",
        AnomalyType.UPDATE_FREQUENCY_ANOMALY: "Update frequency is abnormal",
        AnomalyType.MOVEMENT_CONSISTENCY_ANOMALY: "Movement pattern is inconsistent",
        AnomalyType.LOW_CONFIDENCE_SEQUENCE: "Multiple low-confidence points in sequence",
        AnomalyType.ACCELERATION_ANOMALY: "Acceleration exceeds normal limits",
        AnomalyType.BEARING_ANOMALY: "Direction change is unusual",
    }
    
    # Spoofing indicator explanations
    SPOOFING_INDICATORS = {
        SpoofingIndicator.TELEPORTATION: "Impossible location change detected",
        SpoofingIndicator.REPEATED_TELEPORT: "Pattern of repeated impossible jumps",
        SpoofingIndicator.IMPOSSIBLE_SPEED: "Speed exceeds physical limits",
        SpoofingIndicator.TIMESTAMP_MANIPULATION: "Timestamp sequence appears altered",
        SpoofingIndicator.ACCURACY_DEGRADATION: "GPS accuracy degraded suspiciously",
        SpoofingIndicator.SYNTHETIC_PATTERN: "Movement appears artificially generated",
        SpoofingIndicator.SESSION_INTEGRITY_VIOLATION: "Session data has integrity issues",
        SpoofingIndicator.CLOCK_DRIFT: "Device clock appears manipulated",
        SpoofingIndicator.PERFECT_LINE_PATH: "Path is unnaturally straight",
        SpoofingIndicator.IDENTICAL_COORDINATES: "Same coordinates repeated suspiciously",
    }


class RiskExplainer:
    """
    Generates explanations for risk scores.
    
    Provides both technical and human-readable explanations.
    """
    
    def __init__(self):
        """Initialize risk explainer."""
        self.templates = ExplanationTemplates()
    
    def explain_location_risk(
        self,
        risk: LocationRiskScore,
        verbose: bool = False,
    ) -> Dict[str, any]:
        """
        Generate explanation for a location risk score.
        
        Args:
            risk: Location risk score
            verbose: Include technical details
            
        Returns:
            Explanation dictionary
        """
        explanation = {
            "summary": self.templates.RISK_LEVELS.get(
                risk.risk_level,
                f"Risk level: {risk.risk_level.value}"
            ),
            "risk_score": risk.risk_score,
            "risk_level": risk.risk_level.value,
            "primary_reasons": [],
            "contributing_factors": [],
        }
        
        # Add primary reasons with explanations
        for reason in risk.reasons[:5]:  # Top 5 reasons
            if reason in [a.value for a in AnomalyType]:
                anomaly_type = AnomalyType(reason)
                explanation["primary_reasons"].append({
                    "factor": reason,
                    "explanation": self.templates.ANOMALY_TYPES.get(
                        anomaly_type,
                        reason.replace("_", " ").title()
                    ),
                })
            elif reason in [s.value for s in SpoofingIndicator]:
                indicator = SpoofingIndicator(reason)
                explanation["primary_reasons"].append({
                    "factor": reason,
                    "explanation": self.templates.SPOOFING_INDICATORS.get(
                        indicator,
                        reason.replace("_", " ").title()
                    ),
                })
            else:
                explanation["primary_reasons"].append({
                    "factor": reason,
                    "explanation": reason.replace("_", " ").title(),
                })
        
        # Add contributing factors with weights
        for contributor in sorted(
            risk.contributors,
            key=lambda c: c.weighted_contribution,
            reverse=True
        ):
            factor_info = {
                "name": contributor.factor,
                "contribution": f"{contributor.weighted_contribution:.1%}",
                "description": contributor.description,
            }
            
            if verbose:
                factor_info["weight"] = contributor.weight
                factor_info["raw_score"] = contributor.score
            
            explanation["contributing_factors"].append(factor_info)
        
        # Generate narrative
        explanation["narrative"] = self._generate_location_narrative(risk)
        
        return explanation
    
    def explain_session_risk(
        self,
        risk: SessionRiskScore,
        verbose: bool = False,
    ) -> Dict[str, any]:
        """
        Generate explanation for a session risk score.
        
        Args:
            risk: Session risk score
            verbose: Include technical details
            
        Returns:
            Explanation dictionary
        """
        explanation = {
            "summary": self._generate_session_summary(risk),
            "risk_score": risk.risk_score,
            "risk_level": risk.risk_level.value,
            "statistics": {
                "total_points": risk.total_points,
                "high_risk_points": risk.high_risk_points,
                "anomalous_points": risk.anomalous_points,
                "high_risk_rate": f"{(risk.high_risk_points / risk.total_points * 100):.1f}%" if risk.total_points > 0 else "0%",
            },
            "primary_concerns": [],
            "contributing_factors": [],
        }
        
        # Add primary concerns
        for reason in risk.reasons[:5]:
            explanation["primary_concerns"].append({
                "concern": reason,
                "explanation": reason.replace("_", " ").title(),
            })
        
        # Add contributing factors
        for contributor in sorted(
            risk.contributors,
            key=lambda c: c.weighted_contribution,
            reverse=True
        ):
            factor_info = {
                "name": contributor.factor,
                "contribution": f"{contributor.weighted_contribution:.1%}",
                "description": contributor.description,
            }
            
            if verbose:
                factor_info["weight"] = contributor.weight
                factor_info["raw_score"] = contributor.score
            
            explanation["contributing_factors"].append(factor_info)
        
        # Add flagged events summary
        if risk.flagged_event_ids:
            explanation["flagged_events"] = {
                "count": len(risk.flagged_event_ids),
                "event_ids": risk.flagged_event_ids[:5],  # First 5
                "note": "These events have the highest risk scores in this session",
            }
        
        # Generate narrative
        explanation["narrative"] = self._generate_session_narrative(risk)
        
        return explanation
    
    def _generate_location_narrative(self, risk: LocationRiskScore) -> str:
        """Generate a narrative explanation for location risk."""
        parts = []
        
        # Overall assessment
        parts.append(self.templates.RISK_LEVELS[risk.risk_level])
        
        # Key factors
        if risk.confidence_factor > 0.5:
            parts.append(f"Data confidence is below normal ({(1-risk.confidence_factor)*100:.0f}%).")
        
        if risk.anomaly_factor > 0.5:
            parts.append("Anomaly detection flagged this location.")
        
        if risk.spoofing_factor > 0.3:
            parts.append(f"Potential spoofing detected ({risk.spoofing_factor*100:.0f}% likelihood).")
        
        if risk.baseline_factor > 0.5:
            parts.append("Behavior deviates significantly from learned baseline.")
        
        return " ".join(parts)
    
    def _generate_session_summary(self, risk: SessionRiskScore) -> str:
        """Generate a summary for session risk."""
        if risk.risk_level == RiskLevel.MINIMAL:
            return "Session appears normal with no significant risk indicators."
        elif risk.risk_level == RiskLevel.LOW:
            return f"Session has minor concerns. {risk.high_risk_points} of {risk.total_points} points flagged."
        elif risk.risk_level == RiskLevel.MODERATE:
            return f"Session has moderate risk. {risk.high_risk_points} high-risk points detected."
        elif risk.risk_level == RiskLevel.ELEVATED:
            return f"Session has elevated risk. Review recommended for {risk.anomalous_points} anomalous points."
        elif risk.risk_level == RiskLevel.HIGH:
            return f"Session has high risk. Significant anomalies or spoofing indicators detected."
        else:
            return f"Session has critical risk. Immediate review required."
    
    def _generate_session_narrative(self, risk: SessionRiskScore) -> str:
        """Generate a narrative explanation for session risk."""
        parts = [self._generate_session_summary(risk)]
        
        if risk.max_location_risk > 0.8:
            parts.append(f"Maximum location risk reached {risk.max_location_risk*100:.0f}%.")
        
        if risk.avg_location_risk > 0.4:
            parts.append(f"Average risk across all points is elevated ({risk.avg_location_risk*100:.0f}%).")
        
        if risk.spoofing_likelihood > 0.3:
            parts.append(f"Session-level spoofing likelihood: {risk.spoofing_likelihood*100:.0f}%.")
        
        if risk.integrity_score < 0.9:
            parts.append(f"Data integrity concerns detected (score: {risk.integrity_score*100:.0f}%).")
        
        return " ".join(parts)


class ExplainabilityEngine:
    """
    Main explainability engine for Phase 2.
    
    Provides unified explanation generation for all
    Phase 2 detection and scoring outputs.
    """
    
    def __init__(self):
        """Initialize explainability engine."""
        self.risk_explainer = RiskExplainer()
    
    def explain_anomaly(
        self,
        anomaly: AnomalyResult,
        verbose: bool = False,
    ) -> Dict[str, any]:
        """
        Generate explanation for anomaly detection result.
        
        Args:
            anomaly: Anomaly detection result
            verbose: Include technical details
            
        Returns:
            Explanation dictionary
        """
        explanation = {
            "is_anomalous": anomaly.is_anomalous,
            "anomaly_score": anomaly.anomaly_score,
            "signals": [],
        }
        
        # Add signal explanations
        for signal in sorted(
            anomaly.signals,
            key=lambda s: s.severity,
            reverse=True
        ):
            signal_info = {
                "type": signal.anomaly_type.value,
                "severity": f"{signal.severity*100:.0f}%",
                "description": signal.description,
            }
            
            if signal.z_score is not None:
                signal_info["statistical_deviation"] = f"{abs(signal.z_score):.1f} standard deviations"
            
            if signal.expected_range is not None:
                signal_info["expected_range"] = f"{signal.expected_range[0]:.2f} - {signal.expected_range[1]:.2f}"
                if signal.observed_value is not None:
                    signal_info["observed_value"] = f"{signal.observed_value:.2f}"
            
            if verbose:
                signal_info["confidence"] = signal.confidence
                signal_info["contribution_weight"] = signal.contribution_weight
            
            explanation["signals"].append(signal_info)
        
        # Generate summary
        if anomaly.is_anomalous:
            primary = anomaly.primary_anomaly
            if primary:
                template = ExplanationTemplates.ANOMALY_TYPES.get(
                    primary,
                    primary.value.replace("_", " ").title()
                )
                explanation["summary"] = f"Anomaly detected: {template}"
            else:
                explanation["summary"] = "Anomaly detected with multiple indicators"
        else:
            explanation["summary"] = "No significant anomalies detected"
        
        return explanation
    
    def explain_spoofing(
        self,
        spoofing: SpoofingResult,
        verbose: bool = False,
    ) -> Dict[str, any]:
        """
        Generate explanation for spoofing detection result.
        
        Args:
            spoofing: Spoofing detection result
            verbose: Include technical details
            
        Returns:
            Explanation dictionary
        """
        explanation = {
            "scope": spoofing.scope,
            "spoofing_likelihood": f"{spoofing.spoofing_likelihood*100:.0f}%",
            "tampering_likelihood": f"{spoofing.tampering_likelihood*100:.0f}%",
            "session_integrity": f"{spoofing.session_integrity_score*100:.0f}%",
            "requires_attention": spoofing.requires_attention,
            "indicators": [],
        }
        
        # Add indicator explanations
        for indicator in sorted(
            spoofing.indicators,
            key=lambda i: i.likelihood,
            reverse=True
        ):
            indicator_info = {
                "type": indicator.indicator.value,
                "likelihood": f"{indicator.likelihood*100:.0f}%",
                "description": indicator.description,
            }
            
            if verbose and indicator.evidence:
                indicator_info["evidence"] = indicator.evidence
            
            explanation["indicators"].append(indicator_info)
        
        # Generate summary
        if spoofing.requires_attention:
            if spoofing.spoofing_likelihood > spoofing.tampering_likelihood:
                explanation["summary"] = f"GPS spoofing likely ({spoofing.spoofing_likelihood*100:.0f}%)"
            else:
                explanation["summary"] = f"Data tampering likely ({spoofing.tampering_likelihood*100:.0f}%)"
        else:
            explanation["summary"] = "No significant spoofing or tampering detected"
        
        return explanation
    
    def explain_location_risk(
        self,
        risk: LocationRiskScore,
        verbose: bool = False,
    ) -> Dict[str, any]:
        """Generate explanation for location risk."""
        return self.risk_explainer.explain_location_risk(risk, verbose)
    
    def explain_session_risk(
        self,
        risk: SessionRiskScore,
        verbose: bool = False,
    ) -> Dict[str, any]:
        """Generate explanation for session risk."""
        return self.risk_explainer.explain_session_risk(risk, verbose)
    
    def generate_full_explanation(
        self,
        location_risk: LocationRiskScore,
        anomaly: Optional[AnomalyResult],
        spoofing: Optional[SpoofingResult],
        verbose: bool = False,
    ) -> Dict[str, any]:
        """
        Generate comprehensive explanation for all detection results.
        
        Args:
            location_risk: Location risk score
            anomaly: Anomaly detection result
            spoofing: Spoofing detection result
            verbose: Include technical details
            
        Returns:
            Comprehensive explanation dictionary
        """
        explanation = {
            "timestamp": datetime.now().isoformat(),
            "event_id": location_risk.event_id,
            "overall_risk": self.explain_location_risk(location_risk, verbose),
        }
        
        if anomaly:
            explanation["anomaly_analysis"] = self.explain_anomaly(anomaly, verbose)
        
        if spoofing:
            explanation["spoofing_analysis"] = self.explain_spoofing(spoofing, verbose)
        
        # Generate high-level summary
        summaries = [explanation["overall_risk"]["narrative"]]
        
        if anomaly and anomaly.is_anomalous:
            summaries.append(explanation["anomaly_analysis"]["summary"])
        
        if spoofing and spoofing.requires_attention:
            summaries.append(explanation["spoofing_analysis"]["summary"])
        
        explanation["executive_summary"] = " ".join(summaries)
        
        return explanation
