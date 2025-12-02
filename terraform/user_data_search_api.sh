#!/bin/bash
set -euo pipefail

apt-get update -y
apt-get install -y docker.io
systemctl enable docker
systemctl start docker

docker pull ${container_image}
docker rm -f search-api || true
docker run -d --name search-api --restart unless-stopped -p ${app_port}:80 ${container_image}
