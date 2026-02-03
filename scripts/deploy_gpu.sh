#!/bin/bash
# =============================================================================
# Serendipity Phase 3 - GPU Instance Deployment Script
# =============================================================================
# Run this on the EC2 GPU instance after uploading the codebase

set -e

echo "=============================================="
echo "Serendipity Phase 3 - GPU Deployment"
echo "=============================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# -----------------------------------------------------------------------------
# STEP 1: Verify GPU
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Step 1: Verifying GPU...${NC}"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    echo -e "${GREEN}✓ GPU verified${NC}"
else
    echo -e "${RED}✗ nvidia-smi not found${NC}"
    exit 1
fi

# -----------------------------------------------------------------------------
# STEP 2: Verify Docker + NVIDIA Runtime
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Step 2: Verifying Docker...${NC}"

if command -v docker &> /dev/null; then
    docker --version
    echo -e "${GREEN}✓ Docker installed${NC}"
else
    echo -e "${RED}✗ Docker not found. Installing...${NC}"
    sudo apt-get update
    sudo apt-get install -y docker.io
    sudo systemctl start docker
    sudo usermod -aG docker $USER
fi

# Check NVIDIA runtime
if docker info 2>/dev/null | grep -q nvidia; then
    echo -e "${GREEN}✓ NVIDIA runtime available${NC}"
else
    echo -e "${YELLOW}⚠ Installing NVIDIA Container Toolkit...${NC}"
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo -e "${GREEN}✓ NVIDIA Container Toolkit installed${NC}"
fi

# Test GPU in Docker
echo -e "\n${YELLOW}Testing GPU in Docker...${NC}"
sudo docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ GPU accessible from Docker${NC}"
else
    echo -e "${RED}✗ GPU not accessible from Docker${NC}"
    exit 1
fi

# -----------------------------------------------------------------------------
# STEP 3: Setup Project Directory
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Step 3: Setting up project...${NC}"

PROJECT_DIR="/home/ubuntu/serendipity-ai"
cd $PROJECT_DIR

# Create Python virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip -q

# Install dependencies (skip GPU packages if not needed for API)
echo "Installing Python dependencies..."
pip install -r requirements.txt -q 2>/dev/null || pip install fastapi uvicorn pydantic pydantic-settings httpx redis aiomysql sqlalchemy structlog python-dotenv boto3 -q

echo -e "${GREEN}✓ Python environment ready${NC}"

# -----------------------------------------------------------------------------
# STEP 4: Create Model Repository
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Step 4: Setting up Triton model repository...${NC}"

MODEL_REPO="/home/ubuntu/models"
mkdir -p $MODEL_REPO/delay_predictor/1
mkdir -p $MODEL_REPO/threat_classifier/1
mkdir -p $MODEL_REPO/drift_detector/1

# Copy configs if they exist
if [ -d "docker/triton/model_repository" ]; then
    cp docker/triton/model_repository/delay_predictor/config.pbtxt $MODEL_REPO/delay_predictor/ 2>/dev/null || true
    cp docker/triton/model_repository/threat_classifier/config.pbtxt $MODEL_REPO/threat_classifier/ 2>/dev/null || true
    cp docker/triton/model_repository/drift_detector/config.pbtxt $MODEL_REPO/drift_detector/ 2>/dev/null || true
fi

echo -e "${GREEN}✓ Model repository created at $MODEL_REPO${NC}"

# -----------------------------------------------------------------------------
# STEP 5: Create .env file
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Step 5: Creating environment configuration...${NC}"

if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# =============================================================================
# Serendipity Phase 3 - Production Environment
# =============================================================================

# Application
APP_ENV=production
DEBUG=false
API_SECRET_KEY=change-this-in-production

# Database (update with real credentials)
DATABASE_URL=mysql+aiomysql://user:password@localhost:3306/serendipity

# NVIDIA / GPU
CUDA_ENABLED=true
TENSORRT_ENABLED=true
TRITON_GRPC_URL=localhost:8001
TRITON_HTTP_URL=localhost:8000

# External APIs (add your keys)
OPENWEATHERMAP_API_KEY=
TOMTOM_API_KEY=

# AWS / S3
S3_BUCKET=serendipity-models
S3_REGION=us-west-2

# Cost Guardrails
GPU_COST_GUARDRAIL_ENABLED=true
GPU_MAX_MONTHLY_SPEND_USD=800
GPU_MIN_MONTHLY_SPEND_USD=500
EOF
    echo -e "${YELLOW}⚠ Created .env - Please update with real API keys!${NC}"
else
    echo -e "${GREEN}✓ .env already exists${NC}"
fi

# -----------------------------------------------------------------------------
# STEP 6: Pull Triton Image
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Step 6: Pulling Triton Inference Server image...${NC}"

sudo docker pull nvcr.io/nvidia/tritonserver:24.01-py3

echo -e "${GREEN}✓ Triton image ready${NC}"

echo ""
echo "=============================================="
echo -e "${GREEN}Deployment preparation complete!${NC}"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Update .env with your API keys"
echo "2. Run: ./scripts/start_services.sh"
echo ""
