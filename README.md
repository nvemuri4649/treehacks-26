# Cena — Personal Data Privacy in the Age of AI

> **TreeHacks '26** — A suite of tools that protect your personal data from AI exploitation.

Cena provides two core functionalities:

1. **Image & Video Glazing** — A translucent system overlay (macOS) that intercepts images/videos before upload and applies adversarial glazing to defeat deepfake generation and AI inpainting. A small loading indicator and checkmark appear in a translucent bubble, seamlessly replacing the original with a glazed version.

2. **Privacy Shield Agent** — A dual local-cloud agent system with a web-based chat interface. Documents, images, and messages are dereferenced/redacted locally (on your machine) before being sent to cloud LLMs for heavy reasoning. Images uploaded here also get the glazing treatment. Personal information never leaves your device unprotected.

**Key principle:** All personalized information is processed locally (on the SSH DGX Spark) — the local agent, glazing, redaction — before being dereferenced/glazed and sent to the cloud.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         YOUR MACHINE (DGX Spark / SSH)                      │
│                                                                             │
│  ┌──────────────────────┐    ┌──────────────────────────────────────────┐   │
│  │  GlazeGuard (macOS)  │    │  Privacy Shield Local Guardian          │   │
│  │  ├─ Clipboard monitor│    │  ├─ Nemotron (vLLM, local)              │   │
│  │  ├─ Translucent HUD  │    │  ├─ PII Redactor (placeholder)         │   │
│  │  └─ Auto-glaze on    │    │  ├─ Image Glazer (→ DiffusionGuard)    │   │
│  │     upload            │    │  └─ Re-referencing engine              │   │
│  └────────┬─────────────┘    └──────────────┬──────────────────────────┘   │
│           │                                  │                              │
│  ┌────────▼─────────────┐                    │ sanitized only               │
│  │  Agent Loop          │                    │                              │
│  │  ├─ Glaze            │                    │                              │
│  │  ├─ Generate deepfake│    ┌───────────────▼──────────────────────────┐   │
│  │  ├─ Judge (Claude)   │    │  DiffusionGuard Server (Flask :5000)    │   │
│  │  └─ Increase eps     │    │  ├─ PGD adversarial perturbation        │   │
│  └──────────────────────┘    │  ├─ SD Inpainting pipeline              │   │
│                              │  └─ Fawkes facial recognition cloak     │   │
│                              └─────────────────────────────────────────┘   │
└──────────────────────────────────────────┬──────────────────────────────────┘
                                           │ redacted / glazed data only
                                           ▼
                              ┌──────────────────────────┐
                              │  Cloud Relay              │
                              │  ├─ Claude (Agent SDK)    │
                              │  └─ OpenAI GPT (API)      │
                              └──────────────────────────┘
```

## Project Structure

```
.
├── README.md                    # This file
├── GlazeGuard/                  # macOS menu-bar app (translucent overlay UI)
│   ├── GlazeGuard/
│   │   ├── App/                 # App entry point, state, menu bar
│   │   ├── UI/                  # Overlay window, progress view, approval dialog
│   │   ├── Models/              # Settings, backend config, glazing jobs
│   │   └── Services/            # Backend comms, glazing queue, pasteboard monitor
│   ├── Package.swift
│   └── README.md
├── agents/                      # Privacy Shield multi-agent system
│   ├── local_guardian/          # Agent 1: runs locally (Nemotron via vLLM)
│   │   ├── agent.py             # Orchestrator with tool-calling loop
│   │   ├── redactor.py          # PII redaction (placeholder — implement later)
│   │   ├── image_transformer.py # Image privacy transform (→ uses glazing)
│   │   ├── mapping_store.py     # Token ↔ PII bidirectional map
│   │   ├── rereferencer.py      # Restore PII tokens in cloud responses
│   │   └── nemotron_client.py   # vLLM OpenAI-compatible client
│   └── cloud_relay/             # Agent 2: cloud LLM router
│       ├── agent.py             # Routes to Claude or GPT
│       ├── claude_client.py     # Claude Agent SDK + Anthropic API
│       └── openai_client.py     # OpenAI Chat Completions API
├── frontend/                    # Web-based Privacy Shield chat UI
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── server/
│   ├── app.py                   # Flask GPU server (DiffusionGuard + Fawkes)
│   ├── main.py                  # FastAPI server (Privacy Shield chat)
│   ├── routes.py                # WebSocket + REST endpoints for chat
│   └── requirements.txt         # GPU server dependencies
├── client/
│   ├── glaze.py                 # CLI to protect images
│   ├── agent_loop.py            # Adversarial glaze → generate → judge loop
│   ├── rater_agent.py           # Claude vision-based deepfake rater
│   ├── backends.py              # Backend resolver (reads backends.json)
│   └── requirements.txt         # Client dependencies
├── config/
│   ├── settings.py              # Privacy Shield configuration
│   └── __init__.py
├── backends.json                # Named GPU backends config
├── .env.example                 # Environment variable template
├── deploy.sh                    # GPU server deployment script
└── requirements.txt             # Privacy Shield Python dependencies
```

## Feature 1: Image & Video Glazing (GlazeGuard)

The GlazeGuard macOS menu-bar app provides a **translucent overlay** that appears when images or videos are being uploaded. It:

- Monitors the clipboard for images
- Shows a translucent bubble with loading/checkmark status
- Applies DiffusionGuard adversarial glazing (low eps for visual similarity, auto-increased before model inference)
- Replaces the original with the glazed version seamlessly

The **Agent Loop** augments glazing with a local multiagent pipeline:
1. **Glaze** the image with adversarial perturbations
2. **Generate** a deepfake from the protected image
3. **Judge** the deepfake quality with Claude's vision
4. **Iterate** — increase glazing strength until deepfakes are defeated

### Quick Start — Glazing Server (on DGX Spark)

```bash
# SSH into your DGX Spark
ssh nikhil@spark-abcd.local

# Start the glazing server
docker exec -it diffguard bash -c 'cd /workspace/project/server && python app.py'
```

### Quick Start — GlazeGuard App (macOS)

```bash
cd GlazeGuard
make build    # Build the macOS app
make run      # Run it
```

### CLI Glazing

```bash
# From your laptop
pip install requests pillow
python client/glaze.py --image photo.png --mask mask.png --backend gx10
```

### Agent Loop

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python client/agent_loop.py \
  --image face.png --mask mask.png \
  --prompt "a person in jail" \
  --threshold 6 --backend gx10
```

## Feature 2: Privacy Shield Chat (Web UI)

A web-based chat interface for talking to cloud AI models with full privacy protection. The local Nemotron agent handles all PII before anything reaches the cloud.

### Quick Start — Local Agent (on DGX Spark)

```bash
# Start Nemotron via vLLM on the DGX Spark
ssh nikhil@spark-abcd.local
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 --port 8001 --trust-remote-code
```

### Quick Start — Privacy Shield Server

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your Anthropic / OpenAI keys

# Start the server
python -m server.main
```

### Open the Chat

Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

**How it works:**
1. You type a message or upload an image/document
2. **Nemotron (local)** redacts PII from text + glazes images — all on your machine
3. Sanitized data goes to **Cloud Relay** (Claude Agent SDK / OpenAI GPT)
4. Cloud response returns with redaction tokens
5. **Nemotron (local)** restores your original information and shows the response

## GPU Backend Configuration

Edit `backends.json` to configure GPU backends for glazing:

```json
{
    "backends": {
        "gx10": {
            "url": "http://spark-abcd.local:5000",
            "type": "ssh-docker",
            "ssh": "nikhil@spark-abcd.local"
        },
        "runpod": {
            "url": "https://{POD_ID}-5000.proxy.runpod.net",
            "type": "runpod"
        },
        "local": {
            "url": "http://localhost:5000",
            "type": "local"
        }
    },
    "default": "gx10"
}
```

Environment variables:
```bash
RUNPOD_POD_ID=abc123         # RunPod pod ID
DIFFGUARD_BACKEND=gx10       # Default glazing backend
DIFFGUARD_SERVER=http://...  # Direct URL override
NEMOTRON_ENDPOINT=http://spark-abcd.local:8001/v1   # Local Nemotron
ANTHROPIC_API_KEY=sk-ant-... # For cloud Claude
OPENAI_API_KEY=sk-...        # For cloud GPT
```

## API Endpoints

### Glazing Server (Flask, port 5000)

| Method | Endpoint        | Description                                     |
|--------|----------------|-------------------------------------------------|
| GET    | `/health`      | Server status + GPU info                        |
| POST   | `/protect`     | Protect an image (multipart: `image` + `mask`)  |
| POST   | `/test-inpaint`| Run inpainting to test protection effectiveness |
| POST   | `/fawkes`      | Fawkes facial recognition cloaking              |

### Privacy Shield Server (FastAPI, port 8000)

| Method | Endpoint               | Description                    |
|--------|------------------------|--------------------------------|
| GET    | `/health`              | Service healthcheck            |
| POST   | `/api/session`         | Create a new chat session      |
| DELETE | `/api/session/{id}`    | Delete a session               |
| WS     | `/ws/{session_id}`     | WebSocket real-time chat       |

## Placeholders (To Implement)

- **PII Redactor** (`agents/local_guardian/redactor.py`): Currently passes text through unchanged. Implement NER/regex-based PII detection.
- **Image Dereferencer** (`agents/local_guardian/image_transformer.py`): Currently passes images through unchanged. Will be connected to the glazing pipeline.

## References

- [DiffusionGuard (ICLR 2025)](https://arxiv.org/abs/2410.05694)
- [NVIDIA Nemotron](https://build.nvidia.com/nvidia/NVIDIA-Nemotron-Nano-9B-v2)
- [Claude Agent SDK](https://docs.anthropic.com/en/docs/agents-and-tools/claude-agent-sdk)
- [ASUS Ascent GX10](https://www.asus.com/us/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/)
