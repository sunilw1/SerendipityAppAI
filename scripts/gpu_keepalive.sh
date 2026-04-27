#!/bin/bash
# =============================================================================
# GPU Keep-Alive Script
# =============================================================================
# Runs periodic inference to maintain GPU utilization
# This ensures we meet the $500/month NVIDIA spend requirement

echo "Starting GPU keep-alive service..."
echo "This will run inference every 5 minutes to maintain GPU utilization."

API_URL="http://localhost:8000/api/v1"
API_KEY="dev-secret-key-change-in-production"

# Counter for logging
count=0

while true; do
    count=$((count + 1))
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] Running inference batch #$count..."
    
    # Run delay predictions for multiple users
    for user_id in 1 2 3 4 5; do
        curl -s -H "X-API-Key: $API_KEY" \
            "$API_URL/phase3/predict/delay/$user_id?origin_lat=37.77&origin_lon=-122.41&destination_lat=37.33&destination_lon=-121.88" \
            > /dev/null 2>&1
    done
    
    # Run threat predictions
    for user_id in 1 2 3 4 5; do
        curl -s -H "X-API-Key: $API_KEY" \
            "$API_URL/phase3/predict/threat/$user_id" \
            > /dev/null 2>&1
    done
    
    # Run drift detection
    for user_id in 1 2 3; do
        curl -s -H "X-API-Key: $API_KEY" \
            "$API_URL/phase3/detect/drift/$user_id" \
            > /dev/null 2>&1
    done
    
    # Log GPU status
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
    gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
    
    echo "[$timestamp] Batch #$count complete. GPU Util: ${gpu_util}%, Memory: ${gpu_mem}MB"
    
    # Wait 5 minutes before next batch
    sleep 300
done
