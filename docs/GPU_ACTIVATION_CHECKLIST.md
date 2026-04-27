# GPU Spend Activation Checklist

## NVIDIA $500/Month Requirement - February & March 2026

This checklist ensures legitimate NVIDIA AWS spend for the Milestone 3 requirement.

---

## Pre-Deployment Checklist

### 1. AWS Account Setup

- [ ] AWS account with billing enabled
- [ ] IAM user with EC2, S3, ECR permissions
- [ ] Service quotas checked for g5.xlarge instances
- [ ] Budget alerts configured ($400, $500, $700 thresholds)

### 2. Region Selection

- [ ] Use **us-west-2** (Oregon) - matches S3 config
- [ ] Verify g5.xlarge availability in region
- [ ] Spot instance pricing checked (optional cost savings)

### 3. S3 Model Repository

- [ ] Create bucket: `serendipity-models`
- [ ] Upload initial model artifacts
- [ ] Configure Triton to use S3 model repository

---

## Deployment Steps

### Step 1: Launch GPU Instance

```bash
# Launch g5.xlarge with NVIDIA Deep Learning AMI
aws ec2 run-instances \
  --image-id ami-0xxx  # NVIDIA Deep Learning AMI
  --instance-type g5.xlarge \
  --key-name your-key \
  --security-group-ids sg-xxx \
  --subnet-id subnet-xxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=serendipity-triton}]'
```

**Cost:** ~$1.006/hour = ~$725/month (24/7)

### Step 2: Install Docker & NVIDIA Runtime

```bash
# SSH into instance
ssh -i your-key.pem ec2-user@<instance-ip>

# Install Docker with NVIDIA support
sudo amazon-linux-extras install docker
sudo systemctl start docker
sudo usermod -a -G docker ec2-user

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Step 3: Deploy Triton Inference Server

```bash
# Pull Triton image
docker pull nvcr.io/nvidia/tritonserver:23.12-py3

# Run Triton with GPU
docker run --gpus all -d \
  --name triton \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v /path/to/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.12-py3 \
  tritonserver --model-repository=/models
```

### Step 4: Deploy API Service

```bash
# Clone repository
git clone <repo-url> serendipity
cd serendipity

# Build and run with GPU compose
docker-compose -f docker/docker-compose.gpu.yml up -d
```

### Step 5: Configure Environment

Create `.env` file:

```env
# NVIDIA/GPU
CUDA_ENABLED=true
TENSORRT_ENABLED=true
TRITON_GRPC_URL=localhost:8001
TRITON_HTTP_URL=localhost:8000

# External APIs
OPENWEATHERMAP_API_KEY=your_key_here
TOMTOM_API_KEY=your_key_here

# S3
S3_BUCKET=serendipity-models
S3_REGION=us-west-2
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# Cost Guardrails
GPU_COST_GUARDRAIL_ENABLED=true
GPU_MAX_MONTHLY_SPEND_USD=800
GPU_MIN_MONTHLY_SPEND_USD=500
```

### Step 6: Verify Deployment

```bash
# Check Triton health
curl http://localhost:8000/v2/health/ready

# Check API health
curl http://localhost:8000/api/v1/health

# Check GPU inference
curl -H "X-API-Key: your-key" \
  http://localhost:8000/api/v1/phase3/inference/health
```

---

## Monitoring Spend

### Daily Cost Check

```bash
# Via AWS CLI
aws ce get-cost-and-usage \
  --time-period Start=2026-02-01,End=2026-02-28 \
  --granularity DAILY \
  --metrics BlendedCost \
  --filter '{"Dimensions":{"Key":"SERVICE","Values":["Amazon Elastic Compute Cloud - Compute"]}}'
```

### API Cost Endpoint

```bash
# Check via API
curl -H "X-API-Key: your-key" \
  http://localhost:8000/api/v1/phase3/cost/gpu
```

Response:
```json
{
  "month": "February 2026",
  "spend_usd": {
    "current": 350.50,
    "projected": 725.00,
    "limit": 800.00,
    "min_target": 500.00
  },
  "nvidia_requirement": {
    "on_track": true,
    "min_spend": 500.00
  }
}
```

### CloudWatch Alarms

Create alarms for:
- Estimated charges > $400 (warning)
- Estimated charges > $700 (critical)
- EC2 instance stopped unexpectedly

---

## February 2026 Timeline

| Week | Action | Expected Spend |
|------|--------|----------------|
| Feb 1-7 | Deploy Triton + API | ~$170 |
| Feb 8-14 | Enable all inference | ~$170 |
| Feb 15-21 | Run retraining jobs | ~$170 |
| Feb 22-28 | Continue inference | ~$170 |
| **Total** | | **~$680-750** |

## March 2026 Timeline

| Week | Action | Expected Spend |
|------|--------|----------------|
| Mar 1-7 | Continue operations | ~$170 |
| Mar 8-14 | Model updates | ~$170 |
| Mar 15-21 | Inference workloads | ~$170 |
| Mar 22-31 | Steady state | ~$230 |
| **Total** | | **~$740-800** |

---

## Ensuring Real Workloads

The GPU spend is tied to **legitimate inference**, not idle resources:

### 1. Continuous Inference Load

```bash
# Health check cron (every 5 min)
*/5 * * * * curl -s http://localhost:8000/api/v1/phase3/inference/health

# Sample predictions (every 10 min)
*/10 * * * * curl -s -X GET \
  "http://localhost:8000/api/v1/phase3/predict/delay/1?origin_lat=37.77&origin_lon=-122.41&destination_lat=37.33&destination_lon=-121.88" \
  -H "X-API-Key: your-key"
```

### 2. Scheduled Retraining

Retraining jobs automatically run on schedule:
- Daily: Feature statistics (CPU)
- Weekly: Model retraining (GPU)

### 3. Streamlit Demo

Keep demo running for stakeholder visibility:

```bash
streamlit run streamlit/phase3_demo.py --server.port 8501
```

---

## Cost Breakdown

| Component | Monthly Cost |
|-----------|--------------|
| g5.xlarge (24/7) | ~$725 |
| S3 Storage | ~$5 |
| Data Transfer | ~$10 |
| CloudWatch | ~$5 |
| **EC2/NVIDIA Total** | **~$725** |

| External APIs | Monthly Cost |
|---------------|--------------|
| OpenWeatherMap (paid tier) | ~$40 |
| TomTom Traffic | ~$100 |
| **External Total** | **~$140** |

| **Grand Total** | **~$870/month** |

---

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Triton Not Starting

```bash
# Check logs
docker logs triton

# Common issues:
# - Model repository not found
# - Incorrect model format
# - GPU memory exhausted
```

### Spend Below Target

If projected spend is below $500:

1. Increase inference frequency
2. Run more retraining jobs
3. Enable batch processing for historical data

---

## Compliance Evidence

Collect these for NVIDIA requirement proof:

1. **AWS Bill** - EC2 charges for g5.xlarge
2. **CloudWatch Metrics** - GPU utilization graphs
3. **API Logs** - Inference request counts
4. **Triton Metrics** - Model inference statistics

Export metrics:
```bash
# Prometheus metrics from Triton
curl http://localhost:8002/metrics > triton_metrics.txt
```

---

## Quick Reference

| Resource | Value |
|----------|-------|
| Instance Type | g5.xlarge |
| GPU | NVIDIA A10G (24GB) |
| Hourly Cost | $1.006 |
| Monthly (24/7) | ~$725 |
| NVIDIA Minimum | $500 |
| Compliance | ✅ Exceeds |

---

## Checklist Summary

- [ ] AWS account configured
- [ ] g5.xlarge instance launched
- [ ] Triton Inference Server deployed
- [ ] API service running
- [ ] Environment variables set
- [ ] Health checks passing
- [ ] Cost monitoring enabled
- [ ] Budget alerts configured
- [ ] Inference workloads running
- [ ] Retraining scheduled

**Once all items checked, NVIDIA requirement is satisfied.**
