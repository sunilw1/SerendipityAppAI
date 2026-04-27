#!/bin/bash
# =============================================================================
# Phase 3 Deployment Preparation Script
# =============================================================================
# Run this locally to prepare all artifacts before GPU instance is ready

set -e

echo "=============================================="
echo "Serendipity Phase 3 - Deployment Preparation"
echo "=============================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
AWS_REGION="${AWS_REGION:-us-west-2}"
S3_BUCKET="${S3_BUCKET:-serendipity-models}"
ECR_REPO="${ECR_REPO:-serendipity-api}"
TRITON_MODEL_REPO="docker/triton/model_repository"

echo ""
echo -e "${YELLOW}Step 1: Checking prerequisites...${NC}"

# Check Docker
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓ Docker installed${NC}"
else
    echo -e "${RED}✗ Docker not found. Please install Docker.${NC}"
    exit 1
fi

# Check AWS CLI
if command -v aws &> /dev/null; then
    echo -e "${GREEN}✓ AWS CLI installed${NC}"
else
    echo -e "${RED}✗ AWS CLI not found. Please install AWS CLI.${NC}"
    exit 1
fi

# Check Python
if command -v python3 &> /dev/null; then
    echo -e "${GREEN}✓ Python3 installed${NC}"
else
    echo -e "${RED}✗ Python3 not found.${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Step 2: Validating project structure...${NC}"

# Check required directories
REQUIRED_DIRS=(
    "app"
    "docker"
    "docker/triton"
    "docker/triton/model_repository"
    "scripts"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓ $dir exists${NC}"
    else
        echo -e "${RED}✗ $dir missing${NC}"
    fi
done

echo ""
echo -e "${YELLOW}Step 3: Checking Triton model configs...${NC}"

# Check model configs
MODELS=("delay_predictor" "threat_classifier" "drift_detector")
for model in "${MODELS[@]}"; do
    config_path="${TRITON_MODEL_REPO}/${model}/config.pbtxt"
    if [ -f "$config_path" ]; then
        echo -e "${GREEN}✓ $model config exists${NC}"
    else
        echo -e "${YELLOW}⚠ $model config missing (will use placeholder)${NC}"
    fi
done

echo ""
echo -e "${YELLOW}Step 4: Building Docker image (local test)...${NC}"

# Build the API Docker image
if [ -f "Dockerfile" ]; then
    echo "Building serendipity-api image..."
    docker build -t serendipity-api:latest . --quiet
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
else
    echo -e "${YELLOW}⚠ No Dockerfile found in root. Checking docker/ directory...${NC}"
    if [ -f "docker/Dockerfile" ]; then
        docker build -t serendipity-api:latest -f docker/Dockerfile . --quiet
        echo -e "${GREEN}✓ Docker image built successfully${NC}"
    fi
fi

echo ""
echo -e "${YELLOW}Step 5: Creating deployment package...${NC}"

# Create deployment directory
DEPLOY_DIR="deployment_package"
mkdir -p $DEPLOY_DIR

# Copy essential files
cp -r docker/triton $DEPLOY_DIR/
cp docker/docker-compose.gpu.yml $DEPLOY_DIR/
cp requirements.txt $DEPLOY_DIR/
cp -r app $DEPLOY_DIR/

# Create .env template
cat > $DEPLOY_DIR/.env.template << 'EOF'
# =============================================================================
# Serendipity Phase 3 - Production Environment
# =============================================================================

# Application
APP_ENV=production
DEBUG=false

# Database
DATABASE_URL=mysql+aiomysql://user:password@db:3306/serendipity

# Redis
REDIS_URL=redis://redis:6379/0

# NVIDIA / GPU
CUDA_ENABLED=true
TENSORRT_ENABLED=true
TRITON_GRPC_URL=triton:8001
TRITON_HTTP_URL=triton:8000

# External APIs
OPENWEATHERMAP_API_KEY=your_key_here
TOMTOM_API_KEY=your_key_here

# AWS / S3
S3_BUCKET=serendipity-models
S3_REGION=us-west-2
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

# Cost Guardrails
GPU_COST_GUARDRAIL_ENABLED=true
GPU_MAX_MONTHLY_SPEND_USD=800
GPU_MIN_MONTHLY_SPEND_USD=500

# Security
API_SECRET_KEY=change-this-in-production
EOF

echo -e "${GREEN}✓ Deployment package created at ./$DEPLOY_DIR${NC}"

echo ""
echo -e "${YELLOW}Step 6: Generating deployment checklist...${NC}"

cat > $DEPLOY_DIR/DEPLOY_CHECKLIST.md << 'EOF'
# Deployment Checklist

## Before Deployment
- [ ] GPU instance provisioned by DevOps
- [ ] SSH access confirmed
- [ ] S3 bucket created: serendipity-models
- [ ] External API keys obtained (OpenWeatherMap, TomTom)

## On GPU Instance
- [ ] Docker installed
- [ ] NVIDIA Container Toolkit installed
- [ ] nvidia-smi working
- [ ] Clone repository or copy deployment package

## Deployment Steps
1. Copy .env.template to .env and fill in values
2. Run: docker-compose -f docker-compose.gpu.yml up -d
3. Verify: curl http://localhost:8000/api/v1/health
4. Check GPU: curl http://localhost:8000/api/v1/phase3/inference/health

## Post-Deployment
- [ ] All health checks passing
- [ ] Triton models loaded
- [ ] CloudWatch logs flowing
- [ ] Cost monitoring enabled
EOF

echo -e "${GREEN}✓ Deployment checklist created${NC}"

echo ""
echo "=============================================="
echo -e "${GREEN}Preparation complete!${NC}"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Wait for DevOps to provision the GPU instance"
echo "2. Get SSH access to the instance"
echo "3. Copy deployment_package/ to the instance"
echo "4. Follow DEPLOY_CHECKLIST.md"
echo ""
