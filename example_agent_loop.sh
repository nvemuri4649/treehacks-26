#!/bin/bash
# Example: Run the adversarial agent loop to find optimal glazing strength
#
# This script demonstrates how to use the agent loop to automatically find
# the optimal DiffusionGuard protection strength for an image.
#
# Prerequisites:
#   1. GPU backend running (GX10, RunPod, or local)
#   2. ANTHROPIC_API_KEY environment variable set
#   3. Test image and mask prepared

set -e

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set"
    echo "Get your API key from: https://console.anthropic.com/"
    echo ""
    echo "Then run:"
    echo "  export ANTHROPIC_API_KEY='sk-ant-api03-...'"
    exit 1
fi

# Configuration
IMAGE="${1:-photosofnikhil/nikhil.png}"
MASK="${2:-photosofnikhil/nikhil_mask.png}"
PROMPT="${3:-a person in jail}"
THRESHOLD="${4:-6}"
BACKEND="${5:-gx10}"

echo "=============================================="
echo "ADVERSARIAL AGENT LOOP - EXAMPLE RUN"
echo "=============================================="
echo "Image:     $IMAGE"
echo "Mask:      $MASK"
echo "Prompt:    $PROMPT"
echo "Threshold: $THRESHOLD/10"
echo "Backend:   $BACKEND"
echo "=============================================="
echo ""

# Check if files exist
if [ ! -f "$IMAGE" ]; then
    echo "ERROR: Image not found: $IMAGE"
    echo ""
    echo "Usage: $0 [IMAGE] [MASK] [PROMPT] [THRESHOLD] [BACKEND]"
    echo ""
    echo "Example:"
    echo "  $0 face.png mask.png 'a person in a hospital' 6 gx10"
    exit 1
fi

if [ ! -f "$MASK" ]; then
    echo "ERROR: Mask not found: $MASK"
    exit 1
fi

# Check if backend is reachable
echo "Checking GPU backend..."
if [ "$BACKEND" = "gx10" ]; then
    SERVER_URL="http://spark-abcd.local:5000"
elif [ "$BACKEND" = "local" ]; then
    SERVER_URL="http://localhost:5000"
else
    echo "Using backend: $BACKEND"
    SERVER_URL=""
fi

if [ -n "$SERVER_URL" ]; then
    if ! curl -s "$SERVER_URL/health" > /dev/null 2>&1; then
        echo "WARNING: Cannot reach server at $SERVER_URL"
        echo "Make sure the DiffusionGuard server is running."
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "✓ Server reachable at $SERVER_URL"
    fi
fi

echo ""
echo "Starting agent loop..."
echo "(This may take 10-20 minutes depending on how many iterations are needed)"
echo ""

# Run the agent loop
python3 client/agent_loop.py \
    --image "$IMAGE" \
    --mask "$MASK" \
    --prompt "$PROMPT" \
    --threshold "$THRESHOLD" \
    --backend "$BACKEND" \
    --start-iters 100 \
    --iter-increment 100 \
    --max-iters 800 \
    --max-iterations 8

echo ""
echo "=============================================="
echo "Agent loop complete!"
echo "Check the output directory for results."
echo "=============================================="
