#!/bin/bash
# Run ArXplorer Cloud Frontend
# This script starts the lightweight frontend that connects to the cloud API

set -e

API_ENDPOINT="${1}"
PORT="${2:-5001}"

echo "=== ArXplorer Cloud Frontend ==="
echo ""

# Get API endpoint if not provided
if [ -z "$API_ENDPOINT" ]; then
    echo "Detecting API endpoint from terraform..."
    cd "$(dirname "$0")/../terraform"
    API_ENDPOINT=$(terraform output -raw query_api_endpoint 2>/dev/null || echo "")
    cd - > /dev/null
    
    if [ -z "$API_ENDPOINT" ] || [ "$API_ENDPOINT" == "Query API disabled" ]; then
        echo "Could not auto-detect API endpoint"
        echo "Usage: $0 <api-endpoint> [port]"
        echo "Example: $0 http://3.96.123.45:8080 5001"
        exit 1
    fi
fi

echo "API Endpoint: $API_ENDPOINT"
echo "Frontend Port: $PORT"
echo ""

# Check if frontend dependencies are installed
FRONTEND_DIR="$(dirname "$0")/../frontend"
VENV_PATH="$FRONTEND_DIR/venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_PATH"
fi

# Activate venv and install dependencies
echo "Installing dependencies..."
source "$VENV_PATH/bin/activate"
pip install -q -r "$FRONTEND_DIR/requirements_cloud.txt"

# Test API connectivity
echo "Testing API connectivity..."
if curl -s -f "$API_ENDPOINT/health" > /dev/null; then
    echo "✓ API is reachable"
else
    echo "⚠ Could not connect to API"
    echo "  Continuing anyway - you can fix this later"
fi

echo ""
echo "============================================================"
echo "Starting frontend server..."
echo "Open your browser to: http://localhost:$PORT"
echo "============================================================"
echo ""

# Set environment variable and run Flask
export ARXPLORER_API_ENDPOINT="$API_ENDPOINT"
export FLASK_APP="app_cloud.py"
export FLASK_ENV="development"

cd "$FRONTEND_DIR"
python3 app_cloud.py
