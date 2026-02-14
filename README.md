# DiffusionGuard Pipeline — TreeHacks '26

Protect images against malicious AI inpainting using [DiffusionGuard](https://github.com/choi403/DiffusionGuard) (ICLR 2025), running on an ASUS Ascent GX10 (NVIDIA Blackwell GB10).

## Architecture

```
Your Laptop (macOS)                          ASUS Ascent GX10 (DGX OS)
┌──────────────────────┐                     ┌──────────────────────────────┐
│                      │  POST /protect      │  NGC PyTorch Container       │
│  client/glaze.py     │ ──────────────────► │                              │
│  client/test_glazing │                     │  server/app.py (Flask :5000) │
│                      │ ◄────────────────── │  ├── DiffusionGuard (PGD)    │
│                      │  protected.png      │  └── SD Inpainting pipeline  │
└──────────────────────┘                     └──────────────────────────────┘
```

## Project Structure

```
.
├── SETUP_GX10.md           # Full GX10 hardware + software setup guide
├── README.md               # This file
├── deploy.sh               # One-command deploy to GX10
├── server/
│   ├── app.py              # Flask API server (runs on GX10)
│   └── requirements.txt    # Server Python dependencies
├── client/
│   ├── glaze.py            # CLI to protect images
│   ├── test_glazing.py     # End-to-end test proving protection works
│   ├── make_test_mask.py   # Generate test masks
│   └── requirements.txt    # Client Python dependencies
└── test_images/            # Put your test images here
```

## Quick Start

### 1. Set up the GX10

Follow [SETUP_GX10.md](SETUP_GX10.md) for the full out-of-box setup. The short version:

```bash
# On the GX10 (via SSH):
docker pull nvcr.io/nvidia/pytorch:26.01-py3
mkdir -p ~/diffusionguard_project
docker run -dt --name diffguard --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 5000:5000 \
  -v ~/diffusionguard_project:/workspace/project \
  nvcr.io/nvidia/pytorch:26.01-py3
```

### 2. Deploy from your laptop

```bash
./deploy.sh <user>@<gx10-host>
# e.g.: ./deploy.sh nikhil@spark-abcd.local
```

### 3. Start the server (on GX10)

```bash
ssh <user>@<gx10-host>
docker exec -it diffguard bash
cd /workspace/project/server
python app.py
# First run downloads SD Inpainting model (~4GB)
# Server starts on http://0.0.0.0:5000
```

### 4. Protect an image (from your laptop)

```bash
pip install requests pillow

# Generate a test mask
python client/make_test_mask.py --output test_images/face_mask.png

# Protect an image
python client/glaze.py \
  --image test_images/face.png \
  --mask test_images/face_mask.png \
  --server http://<gx10-ip>:5000
```

### 5. Test that protection works

```bash
python client/test_glazing.py \
  --image test_images/face.png \
  --mask test_images/face_mask.png \
  --server http://<gx10-ip>:5000 \
  --prompt "a person being arrested"
```

This runs inpainting on both the original and protected images and generates a
side-by-side comparison grid in `test_results/comparison_grid.png`.

## API Endpoints

| Method | Endpoint        | Description                                     |
|--------|----------------|-------------------------------------------------|
| GET    | `/health`      | Server status + GPU info                        |
| POST   | `/protect`     | Protect an image (multipart: `image` + `mask`)  |
| POST   | `/test-inpaint`| Run inpainting to test protection effectiveness |

### POST /protect

```bash
curl -X POST http://<gx10>:5000/protect \
  -F "image=@photo.png" \
  -F "mask=@mask.png" \
  --output protected.png
```

Query params: `iters` (int, default 200) — PGD optimization iterations.

### POST /test-inpaint

```bash
curl -X POST http://<gx10>:5000/test-inpaint \
  -F "image=@protected.png" \
  -F "mask=@mask.png" \
  -G -d "prompt=a person in jail" \
  --output result.png
```

Query params: `prompt` (str), `steps` (int, default 50), `seed` (int, default 42).

## How It Works

1. **DiffusionGuard** adds imperceptible adversarial noise to your image, specifically
   targeting the early denoising steps of diffusion inpainting models.
2. When someone tries to maliciously edit the protected image using SD Inpainting
   (e.g., changing the background to create a fake scene), the model produces
   degraded, unusable output.
3. The protection is robust against different mask shapes and even transfers to
   different model versions (black-box transfer).

## References

- [DiffusionGuard Paper (ICLR 2025)](https://arxiv.org/abs/2410.05694)
- [DiffusionGuard Code](https://github.com/choi403/DiffusionGuard)
- [ASUS Ascent GX10](https://www.asus.com/us/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/)
