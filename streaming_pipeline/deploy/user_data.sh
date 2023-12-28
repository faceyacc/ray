#!/bin/bash

# Set env vars.
echo "export ALPACA_API_KEY=PKCILVMX4ZBPS1R91MIK" >> /etc/environment
echo "export ALPACA_API_SECRET=lsn5QhZS3diOhadmwaoyCreWAm0IwVMndWjvaTGu" >> /etc/environment
echo "export QDRANT_API_KEY=ITwN8gQ2ELhPCQARDEFA0hSeYtfnRODS2DdnZJ_7xDTKpmp53U3diA" >> /etc/environment
echo "export QDRANT_URL=https://a4b41bed-46ce-4a49-891b-218813df70ae.us-east4-0.gcp.cloud.qdrant.io" >> /etc/environment

# Install Docker.
sudo apt update
sudo apt install ca-certificates curl gnupg lsb-release -y
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y

sudo usermod -aG docker ubuntu
newgrp docker

# Restart Docker.
sudo systemctl start docker
sudo systemctl enable docker

# Install AWS CLI.
sudo apt update
sudo apt install awscli -y

# Sleep for 90 seconds to allow the instance to fully initialize.
echo "Sleeping for 90 seconds to allow the instance to fully initialize..."
sleep 90

# Authenticate Docker to the ECR registry.
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 693954202752.dkr.ecr.us-east-1.amazonaws.com/streaming_pipeline

# Pull Docker image from ECR.
echo "Pulling Docker image from ECR: 693954202752.dkr.ecr.us-east-1.amazonaws.com/streaming_pipeline/streaming_pipeline:latest"
docker pull 693954202752.dkr.ecr.us-east-1.amazonaws.com/streaming_pipeline:latest

# Run Docker image.
echo "Running Docker image: 693954202752.dkr.ecr.us-east-1.amazonaws.com/streaming_pipeline/streaming_pipeline:latest"
source /etc/environment && docker run --rm \
    -e BYTEWAX_PYTHON_FILE_PATH=tools.run_real_time:build_flow \
    -e ALPACA_API_KEY=PKCILVMX4ZBPS1R91MIK \
    -e ALPACA_API_SECRET=lsn5QhZS3diOhadmwaoyCreWAm0IwVMndWjvaTGu \
    -e QDRANT_API_KEY=ITwN8gQ2ELhPCQARDEFA0hSeYtfnRODS2DdnZJ_7xDTKpmp53U3diA \
    -e QDRANT_URL=https://a4b41bed-46ce-4a49-891b-218813df70ae.us-east4-0.gcp.cloud.qdrant.io \
    --name streaming_pipeline \
    693954202752.dkr.ecr.us-east-1.amazonaws.com/streaming_pipeline:latest