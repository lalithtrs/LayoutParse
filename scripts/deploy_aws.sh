#!/usr/bin/env bash
# Automated Deployment Script for AWS EC2
# Run this directly on your remote Ubuntu EC2 instance

set -e

echo "=========================================================="
echo " Starting DocuExtract AWS EC2 Automated Deployment"
echo "=========================================================="

echo "[1/4] Optimizing disk space & installing Docker..."
# Ensure full 25GB EBS volume is expanded
sudo growpart /dev/nvme0n1 1 2>/dev/null || sudo growpart /dev/xvda 1 2>/dev/null || true
sudo resize2fs /dev/nvme0n1p1 2>/dev/null || sudo resize2fs /dev/xvda1 2>/dev/null || true

# Clean up failed docker build caches if any
sudo docker system prune -af --volumes 2>/dev/null || true

sudo apt-get update -y
sudo apt-get install -y docker.io docker-compose-v2 git curl || sudo apt-get install -y docker.io docker-compose git curl
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER" || true

echo "[2/4] Preparing directories and model weights..."
mkdir -p models/best_layoutlmv3_sroie logs

if [ -f "$HOME/model.safetensors" ]; then
    echo "Found model.safetensors in home directory. Moving to models/best_layoutlmv3_sroie/..."
    mv "$HOME/model.safetensors" models/best_layoutlmv3_sroie/
fi

echo "[3/4] Building and launching lightweight CPU Docker containers..."
if docker compose version >/dev/null 2>&1; then
    sudo docker compose up -d --build
elif command -v docker-compose >/dev/null 2>&1; then
    sudo docker-compose up -d --build
else
    sudo apt-get install -y docker-compose-v2 || sudo apt-get install -y docker-compose
    sudo docker compose up -d --build
fi

echo "[4/4] Verifying service health..."
sleep 10
for i in {1..8}; do
    if curl -s http://localhost:8000/health | grep -q "healthy"; then
        echo "Service is healthy and responding!"
        break
    else
        echo "Waiting for container initialization ($i/8)..."
        sleep 5
    fi
done

PUBLIC_IP=$(curl -s http://checkip.amazonaws.com || echo "<YOUR-EC2-PUBLIC-IP>")

echo ""
echo "=========================================================="
echo " 🎉 DocuExtract is LIVE on AWS!"
echo "=========================================================="
echo " Interactive Swagger Docs: http://${PUBLIC_IP}:8000/docs"
echo " Standard Web HTTP Docs:   http://${PUBLIC_IP}/docs"
echo " Health Status:            http://${PUBLIC_IP}:8000/health"
echo " Live Drift Telemetry:     http://${PUBLIC_IP}:8000/telemetry"
echo " MLflow Tracking UI:       http://${PUBLIC_IP}:5000"
echo "=========================================================="
