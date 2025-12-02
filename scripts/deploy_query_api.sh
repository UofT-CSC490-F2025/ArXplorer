#!/bin/bash
# Deploy ArXplorer code to Query API EC2 instance

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if terraform outputs are available
if [ ! -d "terraform" ]; then
    echo -e "${RED}Error: Must run from project root directory${NC}"
    exit 1
fi

cd terraform

echo -e "${YELLOW}Getting Query API instance IP from terraform...${NC}"
QUERY_API_IP=$(terraform output -raw query_api_public_ip 2>/dev/null)

if [ -z "$QUERY_API_IP" ] || [ "$QUERY_API_IP" == "Query API disabled" ]; then
    echo -e "${RED}Error: Query API instance not deployed or disabled${NC}"
    echo "Run: terraform apply"
    exit 1
fi

echo -e "${GREEN}✓ Query API IP: $QUERY_API_IP${NC}"

# Get SSH key from terraform
KEY_NAME=$(terraform output -raw key_name 2>/dev/null || echo "arxplorer-key")
SSH_KEY="$HOME/.ssh/${KEY_NAME}.pem"

if [ ! -f "$SSH_KEY" ]; then
    echo -e "${RED}Error: SSH key not found at $SSH_KEY${NC}"
    exit 1
fi

cd ..

echo -e "${YELLOW}Creating deployment package...${NC}"

# Create temporary directory for deployment
DEPLOY_DIR=$(mktemp -d)
trap "rm -rf $DEPLOY_DIR" EXIT

# Copy essential files (but NOT config.yaml - it's already on the server)
echo "Copying files..."
cp -r src "$DEPLOY_DIR/"
cp -r scripts "$DEPLOY_DIR/"

echo -e "${GREEN}✓ Deployment package created${NC}"

echo -e "${YELLOW}Uploading to Query API instance...${NC}"

# Upload files (preserve existing config.yaml on server)
scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -r "$DEPLOY_DIR"/* ubuntu@"$QUERY_API_IP":/tmp/arxplorer_deploy/

# Move files and set permissions (preserve config.yaml)
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no ubuntu@"$QUERY_API_IP" << 'ENDSSH'
# Move src and scripts, but preserve config.yaml
sudo cp -r /tmp/arxplorer_deploy/src /opt/arxplorer/
sudo cp -r /tmp/arxplorer_deploy/scripts /opt/arxplorer/
sudo chown -R ubuntu:ubuntu /opt/arxplorer
rm -rf /tmp/arxplorer_deploy

# Verify config.yaml exists and has correct Milvus host
if [ ! -f /opt/arxplorer/config.yaml ]; then
    echo "ERROR: config.yaml not found on server!"
    exit 1
fi

echo "✓ Using existing config.yaml with Milvus host:"
grep "host:" /opt/arxplorer/config.yaml | head -1
ENDSSH

echo -e "${GREEN}✓ Code uploaded (config.yaml preserved)${NC}"

echo -e "${YELLOW}Starting Query API service...${NC}"

ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no ubuntu@"$QUERY_API_IP" << 'ENDSSH'
sudo systemctl daemon-reload
sudo systemctl enable arxplorer-api
sudo systemctl restart arxplorer-api
sleep 5
sudo systemctl status arxplorer-api --no-pager
ENDSSH

echo -e "${GREEN}✓ Service started${NC}"

# Test health endpoint
echo -e "${YELLOW}Testing health endpoint...${NC}"
sleep 10  # Give service time to start

if curl -f -s "http://$QUERY_API_IP:8080/health" > /dev/null; then
    echo -e "${GREEN}✓ Query API is healthy!${NC}"
    echo ""
    echo -e "${GREEN}Deployment complete!${NC}"
    echo ""
    echo "API endpoint: http://$QUERY_API_IP:8080"
    echo "Health check: curl http://$QUERY_API_IP:8080/health"
    echo "Query example: curl -X POST http://$QUERY_API_IP:8080/api/v1/query -H 'Content-Type: application/json' -d '{\"query\": \"attention is all you need\", \"top_k\": 10}'"
    echo ""
    echo "View logs: ssh -i $SSH_KEY ubuntu@$QUERY_API_IP 'sudo journalctl -u arxplorer-api -f'"
else
    echo -e "${RED}✗ Health check failed${NC}"
    echo "Check logs: ssh -i $SSH_KEY ubuntu@$QUERY_API_IP 'sudo journalctl -u arxplorer-api -n 50'"
    exit 1
fi
