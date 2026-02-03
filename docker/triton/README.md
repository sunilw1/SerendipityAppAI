# Triton Inference Server Configuration

## Overview

This directory contains the NVIDIA Triton Inference Server configuration for
Phase 3 predictive intelligence models.

## Model Repository Structure

```
model_repository/
├── delay_predictor/        # Travel delay prediction model
│   ├── config.pbtxt        # Model configuration
│   └── 1/                  # Model version 1
│       └── model.plan      # TensorRT engine (generated)
├── threat_classifier/      # Threat likelihood prediction
│   ├── config.pbtxt
│   └── 1/
│       └── model.plan
└── drift_detector/         # Behavioral drift detection
    ├── config.pbtxt
    └── 1/
        └── model.plan
```

## Models

### delay_predictor

Predicts travel delay in minutes based on:
- Route distance and historical duration
- Weather impact score
- Traffic delay factor
- Time-of-day features

**Input:** 8-dimensional float32 vector
**Output:** 1-dimensional float32 (delay in minutes)

### threat_classifier

Classifies threat likelihood based on:
- Recent risk score statistics
- Anomaly rate features
- Spoofing signal counts
- Trend indicators

**Input:** 16-dimensional float32 vector
**Output:** 1-dimensional float32 (threat likelihood 0-1)

### drift_detector

Detects behavioral drift by comparing:
- Current behavioral features
- Historical baseline features

**Inputs:** 
- current_features: 32-dimensional float32
- baseline_features: 32-dimensional float32

**Outputs:**
- drift_score: 1-dimensional float32 (0-1)
- drift_signals: 6-dimensional float32 (per-type drift scores)

## Generating TensorRT Engines

Use the optimization script to convert trained models:

```bash
python scripts/optimize_models.py \
    --model delay_predictor \
    --input-model models/delay_predictor.onnx \
    --output docker/triton/model_repository/delay_predictor/1/model.plan \
    --fp16
```

## Running Triton

### Development (Docker Compose)

```bash
docker-compose -f docker/docker-compose.gpu.yml up triton
```

### AWS ECS

See the AWS deployment guide in `docs/aws-deployment.md`.

## Performance Tuning

### Batch Size

The `max_batch_size` and `preferred_batch_size` settings in each config.pbtxt
can be tuned based on workload:

- High throughput: Increase batch sizes
- Low latency: Decrease batch sizes

### Instance Count

For higher concurrency, increase `count` in `instance_group`:

```protobuf
instance_group [
  {
    count: 2  # Run 2 instances on GPU 0
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
```

## Health Checks

Triton provides health endpoints:

- Live: `GET /v2/health/live`
- Ready: `GET /v2/health/ready`
- Model Ready: `GET /v2/models/{model}/ready`

## Metrics

Triton exposes Prometheus metrics at `GET /metrics`:

- `nv_inference_request_success`: Successful inference count
- `nv_inference_request_failure`: Failed inference count
- `nv_inference_exec_count`: Total execution count
- `nv_inference_queue_duration_us`: Queue wait time
- `nv_inference_compute_infer_duration_us`: Inference time
