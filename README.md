# Privacy Shield: Multi-Agent Local Privacy Proxy

A local multi-agent system that acts as a privacy proxy between users and cloud LLMs (Claude / GPT). Personal information is redacted before leaving your machine, and cloud responses are re-referenced locally before being shown to you.

## Architecture

- **Agent 1 — Local Guardian** (NVIDIA Nemotron via vLLM): Runs entirely on your machine. Uses Nemotron's tool-calling capabilities to orchestrate PII redaction, image transformation, and re-referencing of cloud responses. Personal data never leaves the local machine.
- **Agent 2 — Cloud Relay** (Claude Agent SDK + OpenAI GPT): Performs the actual user task in the cloud using only sanitized/redacted data. Claude requests go through the Claude Agent SDK (`query()`) for full agentic capabilities; GPT requests use the OpenAI API directly.

```
User Message
    |
    v
[Nemotron Local Agent via vLLM]         <-- runs on YOUR machine
    |-- redact_text()                    <-- strips PII locally
    |-- transform_image()               <-- anonymises images locally
    |-- send_to_cloud()  ─────────────> [Cloud Relay]
    |       |                               |-- Claude (Agent SDK)
    |       |                               |-- OpenAI GPT (API)
    |       <── cloud response (tokens) ────|
    |-- re_reference_text()              <-- restores PII locally
    v
Final Response to User
```

## Quick Start

### 1. Start the local Nemotron model (vLLM)

```bash
# Requires an NVIDIA GPU and vLLM installed
pip install vllm
vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 --port 8001 --trust-remote-code
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your Anthropic and/or OpenAI API keys
# Verify NEMOTRON_ENDPOINT points to your vLLM server
```

### 4. Run the server

```bash
python -m server.main
```

### 5. Open the chatbot

Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## How It Works

1. You type a message (and/or upload an image) in the chatbot.
2. **Nemotron (local)** redacts personal information from text and transforms images — all on your machine.
3. The sanitized prompt is sent to the **Cloud Relay**, which forwards it to Claude (via Claude Agent SDK) or GPT (via OpenAI API).
4. The cloud response (still containing redaction tokens) is returned locally.
5. **Nemotron (local)** re-references the tokens back to your original information and displays the response.

Your personal data never leaves your machine.

## Customization

- **Redaction algorithm**: Edit `agents/local_guardian/redactor.py` — implement your PII detection logic in the `redact()` function.
- **Image transformation**: Edit `agents/local_guardian/image_transformer.py` — implement your transformation in the `transform()` function.
- **Cloud model selection**: Set `DEFAULT_CLOUD_MODEL` in `.env` or use the model dropdown in the chatbot UI.
- **Local model**: Change `NEMOTRON_MODEL` and `NEMOTRON_ENDPOINT` in `.env` to point to any vLLM-served model.
