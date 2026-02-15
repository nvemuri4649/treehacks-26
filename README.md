# Cena — Personal Data Privacy in the Age of AI

> **TreeHacks '26** — A suite of tools that protect your personal data from AI exploitation.

Cena is a native macOS app with two core features:

1. **Automatic Likeness Encryption** — A translucent system overlay that intercepts images/videos before upload and encrypts your likeness to defeat deepfake generation and AI inpainting. A small loading indicator and checkmark appear in a translucent bubble, seamlessly replacing the original with an encrypted version.

2. **Cena Agent** — A native chat interface for talking to cloud LLMs with full privacy. Documents, images, and messages are dereferenced/redacted locally before being sent to the cloud. Images uploaded here also get likeness encryption. Personal information never leaves your device unprotected.

Both features are native SwiftUI, unified in a single menu-bar app.

**Key principle:** All personalized information is processed locally (on the DGX Spark) — the local agent, encryption, redaction — before being dereferenced/encrypted and sent to the cloud.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         YOUR MACHINE (DGX Spark / SSH)                      │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Cena (native macOS)                                                │   │
│  │  ├─ Automatic Likeness Encryption (translucent overlay)             │   │
│  │  │   ├─ Clipboard monitor → auto-encrypt on copy/upload             │   │
│  │  │   └─ Translucent HUD with progress                              │   │
│  │  └─ Agent Chat (native SwiftUI window)                              │   │
│  │      ├─ WebSocket → FastAPI backend                                 │   │
│  │      └─ Model picker (Claude / GPT)                                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│  ┌───────────────────────────▼──────────────────────────────────────────┐   │
│  │  Cena Backend (FastAPI :8000 + Flask :5000)                         │   │
│  │  ├─ Local Guardian (Nemotron via vLLM)                              │   │
│  │  │   ├─ PII Redactor (placeholder)                                  │   │
│  │  │   ├─ Likeness Encryptor (DiffusionGuard)                         │   │
│  │  │   └─ Re-referencing engine                                       │   │
│  │  └─ DiffusionGuard Server (PGD + SD Inpainting + Fawkes)           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────┬──────────────────────────────────┘
                                           │ redacted / encrypted data only
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
├── Cena/                        # Native macOS app (SwiftUI)
│   └── Cena/
│       ├── App/                 # CenaApp, AppState, MenuBarController
│       ├── UI/                  # AgentChatView, OverlayWindow, GlazingProgressView,
│       │                        #   ApprovalDialog, SettingsView
│       ├── Services/            # AgentWebSocket, BackendService, GlazingQueue,
│       │                        #   MaskGenerator, PasteboardMonitor
│       └── Models/              # ChatMessage, GlazingJob, Settings, BackendConfig
├── agents/
│   ├── local_guardian/          # Nemotron agent: redact, encrypt, re-reference
│   └── cloud_relay/             # Routes sanitized requests to Claude / GPT
├── server/
│   ├── app.py                   # Flask GPU server (encryption + Fawkes)
│   ├── fawkes_modern.py         # Fawkes facial recognition cloaking
│   ├── main.py                  # FastAPI server (agent WebSocket API)
│   └── routes.py                # WebSocket + REST endpoints
├── client/
│   ├── glaze.py                 # CLI image encryption
│   ├── agent_loop.py            # Adversarial encrypt → generate → judge loop
│   ├── rater_agent.py           # Claude vision deepfake rater
│   └── backends.py              # Backend resolver
├── config/settings.py           # All configuration
├── backends.json                # GPU backend definitions
├── deploy.sh                    # Deploy to GX10 / RunPod / SSH
├── .env.example                 # Environment variable template
└── requirements.txt             # Backend Python dependencies
```

## Quick Start

### 1. Deploy Encryption Server (on DGX Spark)

```bash
./deploy.sh gx10 nikhil@spark-abcd.local
ssh nikhil@spark-abcd.local
docker exec -it diffguard bash -c 'cd /workspace/project/server && python app.py'
```

### 2. Start Agent Backend

```bash
# On DGX Spark: start local Nemotron
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 --port 8001 --trust-remote-code

# On your machine:
cp .env.example .env   # add your API keys
pip install -r requirements.txt
python -m server.main
```

### 3. Build & Run Cena

```bash
cd Cena && swift build -c release
.build/release/Cena
```

The shield icon appears in your menu bar with two main actions:

- **Automatic Likeness Encryption**: Copy an image → translucent bubble → encrypted → paste
- **Open Agent Chat** (Cmd+A): Native chat window connected to the privacy agent

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
