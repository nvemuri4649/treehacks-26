#!/usr/bin/env bash
#
# Deploy server code to the ASUS Ascent GX10
# ===========================================
# Usage:
#   ./deploy.sh <user>@<gx10-host>
#   ./deploy.sh nikhil@spark-abcd.local
#   ./deploy.sh nikhil@192.168.1.42
#
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <user>@<gx10-host>"
    echo "Example: $0 nikhil@spark-abcd.local"
    exit 1
fi

TARGET="$1"
REMOTE_DIR="diffusionguard_project"

echo "=== Deploying to $TARGET ==="

# 1. Copy server code
echo "[1/4] Copying server code..."
scp -r server/ "$TARGET:~/$REMOTE_DIR/"

# 2. Clone DiffusionGuard repo on the GX10 (if not already there)
echo "[2/4] Ensuring DiffusionGuard repo exists..."
ssh "$TARGET" "cd ~/$REMOTE_DIR && [ -d DiffusionGuard ] || git clone https://github.com/choi403/DiffusionGuard.git"

# 3. Check if docker container exists
echo "[3/4] Checking Docker container..."
ssh "$TARGET" "docker ps -a --format '{{.Names}}' | grep -q diffguard && echo 'Container exists' || echo 'Container not found — create it with the commands in SETUP_GX10.md'"

# 4. Install dependencies inside container
echo "[4/4] Installing Python dependencies inside container..."
ssh "$TARGET" "docker start diffguard 2>/dev/null; docker exec diffguard pip install -q diffusers==0.24.0 transformers datasets huggingface-hub numpy omegaconf opencv-contrib-python scikit-learn tqdm hydra-core flask pillow"

echo ""
echo "=== Deploy complete! ==="
echo ""
echo "To start the server:"
echo "  ssh $TARGET 'docker exec -it diffguard bash -c \"cd /workspace/project/server && python app.py\"'"
echo ""
echo "Or interactively:"
echo "  ssh $TARGET"
echo "  docker exec -it diffguard bash"
echo "  cd /workspace/project/server"
echo "  python app.py"
