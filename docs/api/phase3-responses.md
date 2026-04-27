# Phase 3 API Response Examples

This document provides sample JSON responses for all Phase 3 API endpoints.

## Table of Contents

- [Delay Prediction](#delay-prediction)
- [Threat Prediction](#threat-prediction)
- [Drift Detection](#drift-detection)
- [Inference Health](#inference-health)
- [Inference Stats](#inference-stats)
- [GPU Cost Status](#gpu-cost-status)
- [API Quota Status](#api-quota-status)

---

## Delay Prediction

### Endpoint
```
GET /api/v1/phase3/predict/delay/{user_id}
```

### Sample Request
```
GET /api/v1/phase3/predict/delay/12345?origin_lat=37.7749&origin_lon=-122.4194&destination_lat=37.3382&destination_lon=-121.8863
```

### Sample Response
```json
{
  "success": true,
  "prediction": {
    "prediction_id": "delay_a1b2c3d4e5f6",
    "user_id": 12345,
    "route_id": "route_abc123",
    "origin": [37.7749, -122.4194],
    "destination": [37.3382, -121.8863],
    "expected_distance_meters": 72450.5,
    "predicted_delay_minutes": 12.5,
    "delay_confidence": 0.82,
    "expected_duration_minutes": 55.0,
    "route_confidence": 0.75,
    "weather_delay_minutes": 3.2,
    "traffic_delay_minutes": 9.3,
    "travel_context": {
      "weather": {
        "condition": "rain",
        "temperature_celsius": 14.5,
        "humidity_percent": 78.0,
        "wind_speed_ms": 5.2,
        "visibility_meters": 8000.0,
        "precipitation_mm": 2.1,
        "travel_impact_score": 0.35,
        "timestamp": "2026-02-02T14:30:00Z",
        "location_lat": 37.5565,
        "location_lon": -122.1528,
        "source": "openweathermap"
      },
      "traffic": {
        "severity": "moderate",
        "current_speed_kmh": 45.0,
        "free_flow_speed_kmh": 72.0,
        "congestion_ratio": 0.625,
        "delay_factor": 1.6,
        "estimated_delay_seconds": 558.0,
        "segment_start": [37.7749, -122.4194],
        "segment_end": [37.3382, -121.8863],
        "segment_length_meters": 72450.5,
        "timestamp": "2026-02-02T14:30:00Z",
        "source": "tomtom"
      },
      "weather_available": true,
      "traffic_available": true,
      "combined_delay_factor": 1.48,
      "travel_risk_score": 0.32,
      "context_confidence": 0.88,
      "factors": [
        "Moderate traffic congestion detected",
        "Weather impact: rain conditions (visibility: 8000m)"
      ],
      "generated_at": "2026-02-02T14:30:00Z",
      "valid_until": "2026-02-02T14:45:00Z"
    },
    "explanation": [
      "Matched known route (seen 15 times)",
      "Traffic contributing 9.3 min delay",
      "Weather contributing 3.2 min delay",
      "Moderate traffic congestion detected"
    ],
    "predicted_at": "2026-02-02T14:30:00Z",
    "model_version": "1.0",
    "inference_time_ms": 45.2
  },
  "error": null
}
```

---

## Threat Prediction

### Endpoint
```
GET /api/v1/phase3/predict/threat/{user_id}
```

### Sample Request
```
GET /api/v1/phase3/predict/threat/12345?time_horizon_hours=24
```

### Sample Response
```json
{
  "success": true,
  "prediction": {
    "prediction_id": "threat_x1y2z3w4",
    "user_id": 12345,
    "threat_likelihood": 0.35,
    "threat_level": "moderate",
    "threat_confidence": 0.78,
    "time_horizon_hours": 24,
    "contributing_factors": [
      "Elevated average risk (0.42)",
      "Risk trend increasing (strength: 0.25)"
    ],
    "factor_weights": {
      "risk_history": 0.35,
      "anomaly_rate": 0.25,
      "spoofing_signals": 0.20,
      "behavioral_drift": 0.10,
      "risk_trend": 0.10
    },
    "recent_risk_score_avg": 0.42,
    "anomaly_rate_7d": 0.08,
    "spoofing_signals_count": 2,
    "drift_detected": false,
    "risk_trend": "increasing",
    "trend_strength": 0.25,
    "explanation": "Moderate threat indicators detected. Average risk: 0.42, Anomaly rate: 8.0%. Recommend continued monitoring.",
    "recommendations": [
      "Monitor user activity closely",
      "Review flagged sessions"
    ],
    "predicted_at": "2026-02-02T14:30:00Z",
    "model_version": "1.0",
    "inference_time_ms": 28.5
  },
  "error": null
}
```

---

## Drift Detection

### Endpoint
```
GET /api/v1/phase3/detect/drift/{user_id}
```

### Sample Request
```
GET /api/v1/phase3/detect/drift/12345?time_windows=7,30,90
```

### Sample Response
```json
{
  "success": true,
  "result": {
    "result_id": "drift_m1n2o3p4",
    "user_id": 12345,
    "drift_detected": true,
    "drift_severity": 0.28,
    "primary_drift_type": "speed_range",
    "concern_level": "minor_change",
    "is_concerning": false,
    "signals": [
      {
        "drift_type": "speed_range",
        "severity": 0.28,
        "confidence": 0.8,
        "divergence_score": 0.22,
        "baseline_value": 12.5,
        "current_value": 15.8,
        "change_percent": 26.4,
        "description": "Speed increased by 26.4% (from 12.5 to 15.8 m/s)"
      },
      {
        "drift_type": "activity_timing",
        "severity": 0.15,
        "confidence": 0.7,
        "divergence_score": 0.12,
        "baseline_value": null,
        "current_value": null,
        "change_percent": null,
        "description": "Activity timing pattern shifted (JS divergence: 0.120). Most changed hours: 8, 17, 18"
      }
    ],
    "window_7d_drift": 0.28,
    "window_30d_drift": 0.18,
    "window_90d_drift": 0.12,
    "profile_snapshots_compared": 8,
    "explanation": "Minor behavioral changes detected (severity: 0.28). Changes appear to be normal adaptation.",
    "details": [
      "Speed increased by 26.4% (from 12.5 to 15.8 m/s)",
      "Activity timing pattern shifted (JS divergence: 0.120). Most changed hours: 8, 17, 18"
    ],
    "analyzed_at": "2026-02-02T14:30:00Z",
    "model_version": "1.0",
    "inference_time_ms": 52.1
  },
  "error": null
}
```

---

## Inference Health

### Endpoint
```
GET /api/v1/phase3/inference/health
```

### Sample Response
```json
{
  "healthy": true,
  "gpu_available": true,
  "triton_connected": true,
  "models_loaded": [
    "delay_predictor",
    "threat_classifier",
    "drift_detector"
  ],
  "issues": [],
  "checked_at": "2026-02-02T14:30:00Z"
}
```

### Sample Response (Unhealthy)
```json
{
  "healthy": false,
  "gpu_available": true,
  "triton_connected": false,
  "models_loaded": [],
  "issues": [
    "Triton connection failed: Connection refused"
  ],
  "checked_at": "2026-02-02T14:30:00Z"
}
```

---

## Inference Stats

### Endpoint
```
GET /api/v1/phase3/inference/stats
```

### Sample Response
```json
{
  "gpu": {
    "gpu_available": true,
    "gpu_name": "NVIDIA A10G",
    "gpu_memory_total_mb": 24576.0,
    "gpu_memory_used_mb": 4250.5,
    "gpu_memory_free_mb": 20325.5,
    "gpu_utilization_percent": 35.2,
    "tensorrt_available": true,
    "triton_connected": true
  },
  "models": {
    "delay_predictor": {
      "model_name": "delay_predictor",
      "model_version": "1",
      "backend": "triton",
      "total_inferences": 15420,
      "avg_inference_time_ms": 4.2,
      "p50_inference_time_ms": 3.8,
      "p95_inference_time_ms": 6.5,
      "p99_inference_time_ms": 12.1,
      "inferences_per_second": 125.5,
      "error_count": 3,
      "error_rate": 0.0002
    },
    "threat_classifier": {
      "model_name": "threat_classifier",
      "model_version": "1",
      "backend": "triton",
      "total_inferences": 8750,
      "avg_inference_time_ms": 3.1,
      "p50_inference_time_ms": 2.8,
      "p95_inference_time_ms": 4.5,
      "p99_inference_time_ms": 8.2,
      "inferences_per_second": 95.0,
      "error_count": 1,
      "error_rate": 0.0001
    }
  },
  "total_inferences": 24170,
  "avg_inference_time_ms": 3.8,
  "cpu_inferences": 0,
  "gpu_inferences": 0,
  "triton_inferences": 24170,
  "is_healthy": true,
  "health_issues": [],
  "collected_at": "2026-02-02T14:30:00Z",
  "uptime_seconds": 345600.0
}
```

---

## GPU Cost Status

### Endpoint
```
GET /api/v1/phase3/cost/gpu
```

### Sample Response
```json
{
  "month": "February 2026",
  "hours": {
    "used": 168.5,
    "remaining": 575.5,
    "limit": 744.0
  },
  "spend_usd": {
    "current": 169.51,
    "projected": 745.0,
    "limit": 800.0,
    "min_target": 500.0
  },
  "alert": {
    "level": "ok",
    "message": "Spend within normal range"
  },
  "nvidia_requirement": {
    "on_track": true,
    "min_spend": 500.0
  },
  "days_remaining": 26,
  "recommendations": []
}
```

### Sample Response (Warning)
```json
{
  "month": "February 2026",
  "hours": {
    "used": 620.0,
    "remaining": 124.0,
    "limit": 744.0
  },
  "spend_usd": {
    "current": 623.72,
    "projected": 782.0,
    "limit": 800.0,
    "min_target": 500.0
  },
  "alert": {
    "level": "warning",
    "message": "Spend at 77.9% of limit"
  },
  "nvidia_requirement": {
    "on_track": true,
    "min_spend": 500.0
  },
  "days_remaining": 5,
  "recommendations": []
}
```

---

## API Quota Status

### Endpoint
```
GET /api/v1/phase3/cost/api-quotas
```

### Sample Response
```json
{
  "apis": {
    "openweathermap": {
      "name": "OpenWeatherMap",
      "status": "ok",
      "daily": {
        "calls": 245,
        "limit": 1000,
        "percent": 24.5
      },
      "monthly": {
        "calls": 5420,
        "limit": 30000,
        "percent": 18.1
      },
      "rate_limit": {
        "calls_last_minute": 3,
        "limit_per_minute": 60
      },
      "errors_today": 0,
      "avg_latency_ms": 125.5
    },
    "tomtom": {
      "name": "TomTom Traffic",
      "status": "ok",
      "daily": {
        "calls": 892,
        "limit": 2500,
        "percent": 35.7
      },
      "monthly": {
        "calls": 18540,
        "limit": 75000,
        "percent": 24.7
      },
      "rate_limit": {
        "calls_last_minute": 8,
        "limit_per_minute": 100
      },
      "errors_today": 2,
      "avg_latency_ms": 85.2
    }
  },
  "checked_at": "2026-02-02T14:30:00Z"
}
```

---

## Combined Intelligence

### Endpoint
```
GET /api/v1/phase3/intelligence/{user_id}
```

### Sample Response
```json
{
  "user_id": 12345,
  "timestamp": "2026-02-02T14:30:00Z",
  "delay_prediction": null,
  "delay_available": false,
  "threat_prediction": {
    "prediction_id": "threat_x1y2z3w4",
    "user_id": 12345,
    "threat_likelihood": 0.15,
    "threat_level": "low",
    "threat_confidence": 0.82,
    "time_horizon_hours": 24,
    "contributing_factors": [],
    "factor_weights": {
      "risk_history": 0.35,
      "anomaly_rate": 0.25,
      "spoofing_signals": 0.20,
      "behavioral_drift": 0.10,
      "risk_trend": 0.10
    },
    "recent_risk_score_avg": 0.12,
    "anomaly_rate_7d": 0.02,
    "spoofing_signals_count": 0,
    "drift_detected": false,
    "risk_trend": "stable",
    "trend_strength": 0.05,
    "explanation": "Low threat likelihood based on 25 sessions analyzed. No significant risk patterns detected.",
    "recommendations": [
      "Continue standard monitoring"
    ],
    "predicted_at": "2026-02-02T14:30:00Z",
    "model_version": "1.0",
    "inference_time_ms": 18.5
  },
  "threat_available": true,
  "drift_result": null,
  "drift_available": false,
  "overall_risk_level": "low",
  "recommendations": [
    "Continue standard monitoring"
  ]
}
```

---

## Error Responses

### Standard Error Format
```json
{
  "success": false,
  "prediction": null,
  "error": "User profile not found for user_id: 99999"
}
```

### Validation Error
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["query", "origin_lat"],
      "msg": "Field required",
      "input": null
    }
  ]
}
```

### Authentication Error
```json
{
  "detail": "Invalid or missing API key"
}
```
