#!/bin/bash
# User data script for vLLM EC2 instance

set -e

# Log all output
exec > >(tee /var/log/user-data.log)
exec 2>&1

echo "Starting vLLM instance setup..."

# Update system
apt-get update
apt-get upgrade -y

# Install Docker
apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Install NVIDIA drivers and container toolkit
# For Ubuntu 22.04 on g5 instances
ubuntu-drivers autoinstall

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update
apt-get install -y nvidia-docker2

# Restart Docker
systemctl restart docker

# Configure Docker to use NVIDIA runtime by default
cat > /etc/docker/daemon.json <<EOF
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF

systemctl restart docker

# Wait for NVIDIA GPU to be available
echo "Waiting for GPU..."
sleep 30

# Verify GPU is accessible
nvidia-smi || echo "Warning: nvidia-smi failed"

# Create directory for HuggingFace cache
mkdir -p /data/huggingface_cache
chown -R ubuntu:ubuntu /data

# Pull and run vLLM container
echo "Starting vLLM container..."
docker run -d \
    --name arxplorer-vllm \
    --gpus all \
    --restart unless-stopped \
    -p ${vllm_port}:8000 \
    -v /data/huggingface_cache:/root/.cache/huggingface \
    -e HUGGING_FACE_HUB_TOKEN="" \
    vllm/vllm-openai:latest \
    --model ${vllm_model} \
    --port 8000 \
    --host 0.0.0.0 \
    --gpu-memory-utilization ${gpu_memory_util} \
    --max-model-len ${max_model_len} \
    --quantization ${quantization} \
    --kv-cache-dtype ${kv_cache_dtype} \
    ${enforce_eager ? "--enforce-eager" : ""}

# Wait for vLLM to start
echo "Waiting for vLLM to initialize..."
sleep 60

# Health check
for i in {1..10}; do
    if curl -s http://localhost:${vllm_port}/health > /dev/null; then
        echo "vLLM is healthy!"
        break
    fi
    echo "Waiting for vLLM... attempt $i/10"
    sleep 30
done

# Install CloudWatch agent (optional)
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
dpkg -i -E ./amazon-cloudwatch-agent.deb

echo "vLLM setup complete!"
echo "Model: ${vllm_model}"
echo "Port: ${vllm_port}"
echo "GPU Memory Utilization: ${gpu_memory_util}"
echo "Max Model Length: ${max_model_len}"

# Log container status
docker ps
docker logs arxplorer-vllm --tail 50
