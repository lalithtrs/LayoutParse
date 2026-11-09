# AWS Cloud Deployment Guide: DocuExtract API

This guide provides step-by-step instructions to deploy the containerized **DocuExtract** service to AWS for your portfolio.

---

## Architecture on AWS

```text
  Client / Web Browser / Mobile
                │
                ▼
  [AWS Public IP / Route 53 DNS]
                │
                ▼
  [Security Group] Inbound Port 80 (HTTP) & Port 8000 (FastAPI Docs)
                │
                ▼
  [AWS EC2 (t3.large / c6i.large)]
      ├── Docker: docuextract_api (FastAPI + PaddleOCR + LayoutLMv3)
      └── Docker: docuextract_mlflow (MLflow Tracking UI on :5000)
```

---

## Step 1: Launch an AWS EC2 Instance

1. Log into your [AWS Management Console](https://console.aws.amazon.com/).
2. Navigate to **EC2** $\rightarrow$ **Instances** $\rightarrow$ **Launch an instance**.
3. Configure the instance:
   - **Name:** `docuextract-prod`
   - **OS Image:** `Ubuntu Server 22.04 LTS` (or 24.04 LTS)
   - **Instance Type:** `t3.large` (2 vCPUs, 8 GiB Memory) or `c6i.large` (recommended for CPU OCR throughput)
   - **Key Pair:** Select an existing key pair or create a new `.pem` key pair (e.g., `docuextract-key.pem`)
4. Configure **Network Settings (Security Group)**:
   - Allow SSH traffic from your IP (Port 22)
   - Allow HTTP traffic from the internet (Port 80)
   - Add Custom TCP Rule: **Port 8000** from `0.0.0.0/0` (for direct Swagger UI testing)
   - Add Custom TCP Rule: **Port 5000** from your IP (for MLflow dashboard)
5. **Storage:** Set Volume size to **25 GiB** (gp3) to comfortably accommodate Docker images and dependencies.
6. Click **Launch Instance**.

---

## Step 2: Connect to Your Instance & Install Docker

From your local terminal:

```bash
# Set key permissions
chmod 400 path/to/docuextract-key.pem

# SSH into your EC2 instance
ssh -i path/to/docuextract-key.pem ubuntu@<YOUR-EC2-PUBLIC-IP>
```

Once connected to the EC2 terminal, install Docker and Docker Compose in one command:

```bash
sudo apt update && sudo apt install -y docker.io docker-compose git
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu
```
*(Log out and log back in, or run `newgrp docker` so your user can run docker without sudo)*

---

## Step 3: Clone Repository & Deploy Container

```bash
# 1. Clone your GitHub repository
git clone https://github.com/lalithtrs/LayoutParse.git
cd LayoutParse

# 2. Build and start the container stack in background
docker compose -f docker/docker-compose.yml up -d --build
```

---

## Step 4: Transfer Model Weights to the Server

Because `model.safetensors` (500 MB) is ignored by Git, transfer it from your local machine to the EC2 instance:

From your **local machine terminal**:
```bash
scp -i path/to/docuextract-key.pem models/best_layoutlmv3_sroie/model.safetensors ubuntu@<YOUR-EC2-PUBLIC-IP>:~/LayoutParse/models/best_layoutlmv3_sroie/
```

Restart the container to pick up the weights:
```bash
docker restart docuextract_api
```

---

## Step 5: Verify Live Endpoints

Open your web browser and navigate to:
- **Interactive Swagger UI:** `http://<YOUR-EC2-PUBLIC-IP>:8000/docs`
- **Health Check:** `http://<YOUR-EC2-PUBLIC-IP>:8000/health`
- **Telemetry & Drift Metrics:** `http://<YOUR-EC2-PUBLIC-IP>:8000/telemetry`

Test extraction via `curl`:
```bash
curl -X POST "http://<YOUR-EC2-PUBLIC-IP>:8000/extract" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/samples/sample_001.jpg"
```

---

## Cost Optimization & Pausing

When not actively interviewing or demonstrating:
- In AWS Console, click **Instance state** $\rightarrow$ **Stop instance**.
- When stopped, you pay **$0 for compute** (only ~\$2/month for the 25 GB EBS storage disk).
- Whenever you have an interview, click **Start instance** and it is back online in 30 seconds with the exact same IP and setup!
