#!/bin/bash
# =============================================================================
# Serendipity Phase 3 - Start All Services
# =============================================================================

set -e

echo "=============================================="
echo "Starting Serendipity Phase 3 Services"
echo "=============================================="

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_DIR="/home/ubuntu/serendipity-ai"
MODEL_REPO="/home/ubuntu/models"

cd $PROJECT_DIR

# -----------------------------------------------------------------------------
# Start Triton Inference Server
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Starting Triton Inference Server...${NC}"

# Stop existing Triton if running
sudo docker stop triton 2>/dev/null || true
sudo docker rm triton 2>/dev/null || true

# Start Triton with GPU
sudo docker run -d \
    --name triton \
    --gpus all \
    --restart unless-stopped \
    -p 8001:8001 \
    -p 8002:8002 \
    -v $MODEL_REPO:/models \
    nvcr.io/nvidia/tritonserver:24.01-py3 \
    tritonserver --model-repository=/models --strict-model-config=false

echo "Waiting for Triton to start..."
sleep 10

# Check Triton health
if curl -s http://localhost:8001/v2/health/ready > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Triton is healthy${NC}"
else
    echo -e "${YELLOW}⚠ Triton may still be starting (models loading)${NC}"
fi

# -----------------------------------------------------------------------------
# Start FastAPI Backend
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Starting FastAPI Backend...${NC}"

# Activate virtual environment
source venv/bin/activate

# Kill existing uvicorn if running
pkill -f "uvicorn app.main:app" 2>/dev/null || true

# Start API in background
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2 > /var/log/serendipity-api.log 2>&1 &

echo "Waiting for API to start..."
sleep 5

# Check API health
if curl -s http://localhost:8000/api/v1/health/live > /dev/null 2>&1; then
    echo -e "${GREEN}✓ API is healthy${NC}"
else
    echo -e "${YELLOW}⚠ API may still be starting${NC}"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo -e "${GREEN}Services Started!${NC}"
echo "=============================================="
echo ""
echo "Triton Inference Server:"
echo "  - gRPC: localhost:8001"
echo "  - Metrics: localhost:8002"
echo ""
echo "FastAPI Backend:"
echo "  - HTTP: localhost:8000"
echo ""
echo "Health Checks:"
echo "  curl http://localhost:8000/api/v1/health/live"
echo "  curl http://localhost:8001/v2/health/ready"
echo ""
echo "GPU Monitoring:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Logs:"
echo "  tail -f /var/log/serendipity-api.log"
echo "  sudo docker logs -f triton"
echo ""
