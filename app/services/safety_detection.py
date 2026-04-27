"""
Safety Detection Service
=========================

Provides crash detection, fall detection, and battery monitoring.

Crash Detection:
- Analyzes accelerometer data for sudden deceleration patterns
- Considers speed, G-force, and post-impact motion
- Classifies severity from minor to critical

Fall Detection:
- Detects free-fall → impact → stationary pattern
- Considers G-force magnitude, free-fall duration, post-fall state
- Age-aware risk assessment

Battery Monitoring:
- Tracks battery drain rates
- Predicts time to empty
- Triggers alerts for low/critical levels
"""

import math
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from app.core.logging import get_logger
from app.models.phase3_schemas import (
    AccelerometerReading,
    BatteryAnalysisResult,
    BatteryStatus,
    CrashDetectionResult,
    FallDetectionResult,
    ImpactSeverity,
)

logger = get_logger(__name__)

GRAVITY = 9.81

# Crash thresholds (in G-forces)
CRASH_THRESHOLDS = {
    ImpactSeverity.MINOR: 4.0,
    ImpactSeverity.MODERATE: 8.0,
    ImpactSeverity.SEVERE: 15.0,
    ImpactSeverity.CRITICAL: 25.0,
}

# Fall thresholds
FREE_FALL_G_THRESHOLD = 0.4  # Below 0.4G = free-fall
FALL_IMPACT_G_THRESHOLD = 3.0  # Impact after free-fall
STATIONARY_THRESHOLD = 0.3  # G deviation from 1.0 = stationary

# Battery thresholds
BATTERY_LOW_THRESHOLD = 20.0
BATTERY_CRITICAL_THRESHOLD = 10.0


def _compute_g_force(reading: AccelerometerReading) -> float:
    """Compute total G-force magnitude from accelerometer reading."""
    magnitude = math.sqrt(reading.x ** 2 + reading.y ** 2 + reading.z ** 2)
    return magnitude / GRAVITY


def _compute_delta_g(readings: List[AccelerometerReading]) -> List[float]:
    """Compute change in G-force between consecutive readings."""
    deltas = []
    for i in range(1, len(readings)):
        g_prev = _compute_g_force(readings[i - 1])
        g_curr = _compute_g_force(readings[i])
        deltas.append(abs(g_curr - g_prev))
    return deltas


class CrashDetector:
    """Detects vehicle crashes from accelerometer data."""

    def detect(
        self,
        user_id: int,
        readings: List[AccelerometerReading],
        speed_kmh: Optional[float] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> CrashDetectionResult:
        now = datetime.now(timezone.utc)
        detection_id = f"crash_{uuid.uuid4().hex[:12]}"

        g_forces = [_compute_g_force(r) for r in readings]
        peak_g = max(g_forces)
        peak_idx = g_forces.index(peak_g)

        delta_gs = _compute_delta_g(readings)
        max_delta_g = max(delta_gs) if delta_gs else 0

        # Detect crash: sudden spike in G-force
        crash_detected = peak_g >= CRASH_THRESHOLDS[ImpactSeverity.MINOR]

        # Speed context: low-speed bumps are less likely crashes
        if speed_kmh is not None and speed_kmh < 10 and peak_g < CRASH_THRESHOLDS[ImpactSeverity.MODERATE]:
            crash_detected = False

        severity = ImpactSeverity.NONE
        if crash_detected:
            if peak_g >= CRASH_THRESHOLDS[ImpactSeverity.CRITICAL]:
                severity = ImpactSeverity.CRITICAL
            elif peak_g >= CRASH_THRESHOLDS[ImpactSeverity.SEVERE]:
                severity = ImpactSeverity.SEVERE
            elif peak_g >= CRASH_THRESHOLDS[ImpactSeverity.MODERATE]:
                severity = ImpactSeverity.MODERATE
            else:
                severity = ImpactSeverity.MINOR

        # Check post-impact motion
        post_impact_readings = g_forces[peak_idx + 1:] if peak_idx < len(g_forces) - 1 else []
        post_impact_motion = True
        if post_impact_readings:
            avg_post = sum(post_impact_readings) / len(post_impact_readings)
            post_impact_motion = abs(avg_post - 1.0) > STATIONARY_THRESHOLD

        # Impact duration estimation
        impact_indices = [i for i, g in enumerate(g_forces) if g >= CRASH_THRESHOLDS[ImpactSeverity.MINOR]]
        if len(impact_indices) >= 2 and len(readings) >= 2:
            t_start = readings[impact_indices[0]].timestamp
            t_end = readings[impact_indices[-1]].timestamp
            impact_duration_ms = abs((t_end - t_start).total_seconds() * 1000)
        else:
            impact_duration_ms = 0

        # Confidence calculation
        confidence = 0.0
        if crash_detected:
            confidence = min(1.0, 0.5 + (peak_g - CRASH_THRESHOLDS[ImpactSeverity.MINOR]) / 20.0)
            if speed_kmh and speed_kmh > 30:
                confidence = min(1.0, confidence + 0.15)
            if max_delta_g > 5.0:
                confidence = min(1.0, confidence + 0.1)
            if not post_impact_motion:
                confidence = min(1.0, confidence + 0.1)

        location = (lat, lon) if lat is not None and lon is not None else None

        explanation_parts = []
        if crash_detected:
            explanation_parts.append(
                f"Crash detected with peak force of {peak_g:.1f}G ({severity.value} severity)."
            )
            if speed_kmh:
                explanation_parts.append(f"Speed at impact: {speed_kmh:.0f} km/h.")
            if not post_impact_motion:
                explanation_parts.append("No motion detected after impact - potential injury.")
        else:
            explanation_parts.append(
                f"No crash detected. Peak G-force: {peak_g:.1f}G (below {CRASH_THRESHOLDS[ImpactSeverity.MINOR]}G threshold)."
            )

        actions = []
        if crash_detected:
            if severity in (ImpactSeverity.CRITICAL, ImpactSeverity.SEVERE):
                actions.extend([
                    "Contact emergency services immediately",
                    "Notify emergency contacts",
                    "Share GPS location with responders",
                ])
            elif severity == ImpactSeverity.MODERATE:
                actions.extend([
                    "Notify emergency contacts",
                    "Ask user for confirmation of safety",
                ])
            else:
                actions.append("Ask user to confirm they are okay")

        alert_triggered = crash_detected and severity in (
            ImpactSeverity.MODERATE, ImpactSeverity.SEVERE, ImpactSeverity.CRITICAL
        )

        logger.info(
            "crash_detection_complete",
            user_id=user_id,
            detected=crash_detected,
            severity=severity.value,
            peak_g=round(peak_g, 2),
        )

        return CrashDetectionResult(
            detection_id=detection_id,
            user_id=user_id,
            crash_detected=crash_detected,
            severity=severity,
            confidence=round(confidence, 3),
            peak_g_force=round(peak_g, 2),
            impact_duration_ms=round(impact_duration_ms, 1),
            post_impact_motion=post_impact_motion,
            speed_at_impact_kmh=speed_kmh,
            location=location,
            explanation=" ".join(explanation_parts),
            recommended_actions=actions,
            detected_at=now,
            alert_triggered=alert_triggered,
        )


class FallDetector:
    """Detects human falls from accelerometer data."""

    def detect(
        self,
        user_id: int,
        readings: List[AccelerometerReading],
        user_age: Optional[int] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> FallDetectionResult:
        now = datetime.now(timezone.utc)
        detection_id = f"fall_{uuid.uuid4().hex[:12]}"

        g_forces = [_compute_g_force(r) for r in readings]

        # Phase 1: Detect free-fall (G-force drops below threshold)
        free_fall_indices = [i for i, g in enumerate(g_forces) if g < FREE_FALL_G_THRESHOLD]

        # Phase 2: Detect impact spike after free-fall
        impact_after_freefall = False
        impact_g = 0.0
        impact_idx = -1

        for ff_idx in free_fall_indices:
            for j in range(ff_idx + 1, min(ff_idx + 5, len(g_forces))):
                if g_forces[j] >= FALL_IMPACT_G_THRESHOLD:
                    impact_after_freefall = True
                    if g_forces[j] > impact_g:
                        impact_g = g_forces[j]
                        impact_idx = j
                    break

        fall_detected = impact_after_freefall and len(free_fall_indices) > 0

        # Free-fall duration
        free_fall_duration_ms = 0
        if free_fall_indices and len(readings) > 1:
            ff_start = readings[free_fall_indices[0]].timestamp
            ff_end = readings[free_fall_indices[-1]].timestamp
            free_fall_duration_ms = max(0, (ff_end - ff_start).total_seconds() * 1000)

        # Phase 3: Check post-fall stationary state
        post_fall_stationary = False
        stationary_duration = 0
        if fall_detected and impact_idx >= 0:
            post_readings = g_forces[impact_idx + 1:]
            if post_readings:
                avg_post = sum(post_readings) / len(post_readings)
                post_fall_stationary = abs(avg_post - 1.0) < STATIONARY_THRESHOLD
                if post_fall_stationary and impact_idx < len(readings) - 1:
                    t_impact = readings[impact_idx].timestamp
                    t_last = readings[-1].timestamp
                    stationary_duration = max(0, (t_last - t_impact).total_seconds())

        # Severity classification
        severity = ImpactSeverity.NONE
        if fall_detected:
            if impact_g >= 10.0 and post_fall_stationary:
                severity = ImpactSeverity.CRITICAL
            elif impact_g >= 8.0 or (post_fall_stationary and stationary_duration > 10):
                severity = ImpactSeverity.SEVERE
            elif impact_g >= 5.0:
                severity = ImpactSeverity.MODERATE
            else:
                severity = ImpactSeverity.MINOR

        # Age-aware risk adjustment
        if user_age and user_age > 65 and fall_detected:
            if severity == ImpactSeverity.MINOR:
                severity = ImpactSeverity.MODERATE
            elif severity == ImpactSeverity.MODERATE:
                severity = ImpactSeverity.SEVERE

        # Confidence
        confidence = 0.0
        if fall_detected:
            confidence = 0.5
            if free_fall_duration_ms > 50:
                confidence += 0.1
            if impact_g > FALL_IMPACT_G_THRESHOLD * 1.5:
                confidence += 0.15
            if post_fall_stationary:
                confidence += 0.15
            if free_fall_duration_ms > 100 and free_fall_duration_ms < 1000:
                confidence += 0.1
            confidence = min(1.0, confidence)

        location = (lat, lon) if lat is not None and lon is not None else None

        explanation_parts = []
        if fall_detected:
            explanation_parts.append(
                f"Fall detected: free-fall phase ({free_fall_duration_ms:.0f}ms) "
                f"followed by {impact_g:.1f}G impact."
            )
            if post_fall_stationary:
                explanation_parts.append(
                    f"User stationary for {stationary_duration:.0f}s after impact - potential injury."
                )
            if user_age and user_age > 65:
                explanation_parts.append("Elevated risk due to user age.")
        else:
            explanation_parts.append("No fall pattern detected in accelerometer data.")

        actions = []
        if fall_detected:
            if severity in (ImpactSeverity.CRITICAL, ImpactSeverity.SEVERE):
                actions.extend([
                    "Contact emergency services immediately",
                    "Notify emergency contacts",
                    "Share GPS location with responders",
                ])
            elif severity == ImpactSeverity.MODERATE:
                actions.extend([
                    "Notify emergency contacts",
                    "Ask user for confirmation of safety",
                    "Monitor for follow-up activity",
                ])
            else:
                actions.append("Ask user to confirm they are okay")

        alert_triggered = fall_detected and severity in (
            ImpactSeverity.MODERATE, ImpactSeverity.SEVERE, ImpactSeverity.CRITICAL
        )

        logger.info(
            "fall_detection_complete",
            user_id=user_id,
            detected=fall_detected,
            severity=severity.value,
            impact_g=round(impact_g, 2),
        )

        return FallDetectionResult(
            detection_id=detection_id,
            user_id=user_id,
            fall_detected=fall_detected,
            severity=severity,
            confidence=round(confidence, 3),
            free_fall_duration_ms=round(free_fall_duration_ms, 1),
            impact_g_force=round(impact_g, 2),
            post_fall_stationary=post_fall_stationary,
            stationary_duration_seconds=round(stationary_duration, 1),
            location=location,
            explanation=" ".join(explanation_parts),
            recommended_actions=actions,
            detected_at=now,
            alert_triggered=alert_triggered,
        )


class BatteryMonitor:
    """Monitors device battery levels and triggers safety alerts."""

    def analyze(
        self,
        user_id: int,
        battery_level: float,
        is_charging: bool = False,
        drain_rate_per_hour: Optional[float] = None,
        battery_temperature: Optional[float] = None,
    ) -> BatteryAnalysisResult:
        now = datetime.now(timezone.utc)

        if is_charging:
            status = BatteryStatus.CHARGING
        elif battery_level <= BATTERY_CRITICAL_THRESHOLD:
            status = BatteryStatus.CRITICAL
        elif battery_level <= BATTERY_LOW_THRESHOLD:
            status = BatteryStatus.LOW
        else:
            status = BatteryStatus.HEALTHY

        # Estimate remaining time
        hours_remaining = None
        if drain_rate_per_hour and drain_rate_per_hour > 0 and not is_charging:
            hours_remaining = round(battery_level / drain_rate_per_hour, 1)

        is_critical = status == BatteryStatus.CRITICAL

        explanation_parts = []
        if is_charging:
            explanation_parts.append(f"Device is charging at {battery_level:.0f}%.")
        elif is_critical:
            explanation_parts.append(
                f"Battery critically low at {battery_level:.0f}%."
            )
            if hours_remaining:
                explanation_parts.append(
                    f"Estimated {hours_remaining:.1f} hours remaining."
                )
            explanation_parts.append("Safety tracking may be interrupted.")
        elif status == BatteryStatus.LOW:
            explanation_parts.append(f"Battery low at {battery_level:.0f}%.")
            if hours_remaining:
                explanation_parts.append(
                    f"Estimated {hours_remaining:.1f} hours remaining."
                )
        else:
            explanation_parts.append(f"Battery healthy at {battery_level:.0f}%.")
            if hours_remaining:
                explanation_parts.append(
                    f"Estimated {hours_remaining:.1f} hours remaining."
                )

        if battery_temperature and battery_temperature > 45:
            explanation_parts.append(
                f"Warning: Battery temperature elevated ({battery_temperature:.0f}°C)."
            )

        actions = []
        if is_critical:
            actions.extend([
                "Notify designated contacts about low battery",
                "Enable power-saving mode for tracking",
                "Reduce GPS update frequency to conserve battery",
                "User should charge device as soon as possible",
            ])
        elif status == BatteryStatus.LOW:
            actions.extend([
                "Alert user to charge device soon",
                "Consider reducing tracking frequency",
            ])

        alert_triggered = status in (BatteryStatus.LOW, BatteryStatus.CRITICAL) and not is_charging

        logger.info(
            "battery_analysis_complete",
            user_id=user_id,
            level=battery_level,
            status=status.value,
            alert=alert_triggered,
        )

        return BatteryAnalysisResult(
            user_id=user_id,
            battery_level=battery_level,
            status=status,
            estimated_hours_remaining=hours_remaining,
            is_critical=is_critical,
            drain_rate_per_hour=drain_rate_per_hour,
            explanation=" ".join(explanation_parts),
            alert_triggered=alert_triggered,
            recommended_actions=actions,
            analyzed_at=now,
        )


# Singletons
_crash_detector: Optional[CrashDetector] = None
_fall_detector: Optional[FallDetector] = None
_battery_monitor: Optional[BatteryMonitor] = None


def get_crash_detector() -> CrashDetector:
    global _crash_detector
    if _crash_detector is None:
        _crash_detector = CrashDetector()
    return _crash_detector


def get_fall_detector() -> FallDetector:
    global _fall_detector
    if _fall_detector is None:
        _fall_detector = FallDetector()
    return _fall_detector


def get_battery_monitor() -> BatteryMonitor:
    global _battery_monitor
    if _battery_monitor is None:
        _battery_monitor = BatteryMonitor()
    return _battery_monitor
