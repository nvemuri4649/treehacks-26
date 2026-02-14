# Cena — Personal Data Privacy in the Age of AI

> **TreeHacks '26** — A suite of tools that protect your personal data from AI exploitation.

Cena provides two core functionalities:

1. **Image & Video Glazing** — A translucent system overlay (macOS) that intercepts images/videos before upload and applies adversarial glazing to defeat deepfake generation and AI inpainting. A small loading indicator and checkmark appear in a translucent bubble, seamlessly replacing the original with a glazed version.

2. **Privacy Shield Agent** — A dual local-cloud agent system with a web-based chat interface. Documents, images, and messages are dereferenced/redacted locally before being sent to cloud LLMs for heavy reasoning. Images uploaded here also get the glazing treatment. Personal information never leaves your device unprotected.

**Key principle:** All personalized information is processed locally (on the DGX Spark) — the local agent, glazing, redaction — before being dereferenced/glazed and sent to the cloud.

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
├── GlazeGuard/                  # macOS menu-bar app (translucent overlay UI)
│   └── GlazeGuard/              # App/, UI/, Models/, Services/
├── agents/
│   ├── local_guardian/          # Nemotron agent: redact, glaze, re-reference
│   └── cloud_relay/             # Routes sanitized requests to Claude / GPT
├── frontend/                    # Web chat UI (index.html, styles.css, app.js)
├── server/
│   ├── app.py                   # Flask GPU server (DiffusionGuard + Fawkes)
│   ├── fawkes_modern.py         # Fawkes facial recognition cloaking
│   ├── main.py                  # FastAPI server (Privacy Shield chat)
│   └── routes.py                # WebSocket + REST endpoints
├── client/
│   ├── glaze.py                 # CLI image protection
│   ├── agent_loop.py            # Adversarial glaze → generate → judge loop
│   ├── rater_agent.py           # Claude vision deepfake rater
│   └── backends.py              # Backend resolver
├── config/settings.py           # All configuration
├── backends.json                # GPU backend definitions
├── deploy.sh                    # Deploy to GX10 / RunPod / SSH
├── .env.example                 # Environment variable template
└── requirements.txt             # Privacy Shield dependencies
```

## Quick Start

### 1. Deploy Glazing Server (on DGX Spark)

```bash
./deploy.sh gx10 nikhil@spark-abcd.local
ssh nikhil@spark-abcd.local
docker exec -it diffguard bash -c 'cd /workspace/project/server && python app.py'
```

### 2. GlazeGuard App (macOS overlay)

```bash
cd GlazeGuard && swift build -c release    # or: make build && make run
```

Copy an image → translucent bubble appears → image is glazed → paste the protected version.

### 3. Privacy Shield Chat (web UI)

```bash
# On DGX Spark: start local Nemotron
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 --port 8001 --trust-remote-code

# On your machine:
cp .env.example .env   # add your API keys
pip install -r requirements.txt
python -m server.main
# Open http://127.0.0.1:8000
```

### 4. CLI Glazing

```bash
pip install requests pillow
python client/glaze.py --image photo.png --mask mask.png --backend gx10
```

### 5. Agent Loop (auto-optimize glazing strength)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python client/agent_loop.py \
  --image face.png --mask mask.png \
  --prompt "a person in jail" \
  --threshold 6 --backend gx10
```

## Configuration

Copy `.env.example` to `.env` and set:

| Variable | Purpose |
|----------|---------|
| `NEMOTRON_ENDPOINT` | vLLM server URL (default: `http://spark-abcd.local:8001/v1`) |
| `GLAZE_SERVER_URL` | DiffusionGuard Flask server (default: `http://spark-abcd.local:5000`) |
| `ANTHROPIC_API_KEY` | Cloud Claude access |
| `OPENAI_API_KEY` | Cloud GPT access |
| `DIFFGUARD_BACKEND` | Default backend name from `backends.json` |

## Placeholders

- **PII Redactor** (`agents/local_guardian/redactor.py`): Pass-through. Implement NER/regex PII detection.
- **Dereferencer**: Planned — will fully decouple personal identifiers before cloud relay.

## References

- [DiffusionGuard (ICLR 2025)](https://arxiv.org/abs/2410.05694) — adversarial image protection
- [NVIDIA Nemotron](https://build.nvidia.com/nvidia/NVIDIA-Nemotron-Nano-9B-v2) — local LLM
- [Claude Agent SDK](https://docs.anthropic.com/en/docs/agents-and-tools/claude-agent-sdk) — agentic cloud relay
