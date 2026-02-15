# Deepfake Detection Agent

An AI-powered agent that searches the web for potential deepfakes of a person given their photo. Built with the Claude Agent SDK, Bright Data web scraping, and InsightFace face recognition.

## How It Works

1. **Upload** a photo of yourself via the web interface
2. The agent **extracts your facial embedding** using InsightFace (ArcFace, 512-D vector)
3. **Reverse image search** via Bright Data Scraping Browser (Google Lens) finds visually similar images
4. **SERP text/image searches** fan out across Google for deepfake-related queries
5. **Face matching** compares each candidate against your embedding (cosine similarity)
6. **Multi-signal deepfake detection** analyzes matched images:
   - Claude Vision artifact analysis (skin texture, lighting, blending edges)
   - EXIF/metadata inspection (AI generation signatures)
   - Frequency domain analysis (DCT spectral artifacts from GANs)
7. A **threat report** is generated with findings, confidence scores, and source URLs

## Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)
- A [Bright Data account](https://brightdata.com/) with SERP API and Scraping Browser zones
- Playwright browsers installed (`playwright install chromium`)

## Setup

```bash
# Clone and enter the project
cd scraping

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Install Playwright browser
playwright install chromium

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the server
python main.py
```

Then open http://localhost:8000 in your browser.

## Architecture

```
agent/                  # Claude Agent SDK orchestration layer
  orchestrator.py       # Agent setup, tool wiring, run loop
  system_prompt.py      # Specialized deepfake hunting instructions
  tools/                # MCP tool definitions
    reverse_image_search.py
    serp_search.py
    face_analysis.py
    image_download.py
    deepfake_analysis.py
    report_generator.py

core/                   # Core processing engines
  face_engine.py        # InsightFace face detection + embedding
  search_engine.py      # Bright Data API abstraction
  deepfake_detector.py  # Multi-signal deepfake analysis
  image_processor.py    # Image download, validation, dedup

templates/              # Jinja2 HTML templates
static/                 # CSS assets
output/                 # Saved reports and evidence
```

## Configuration

| Variable | Description | Default |
|---|---|---|
| `FACE_MATCH_THRESHOLD` | Cosine distance for same-identity match | `0.35` |
| `DEEPFAKE_FLAG_THRESHOLD` | Minimum score to flag as potential deepfake | `0.6` |
| `MAX_CANDIDATES_PER_SEARCH` | Max images to process per search query | `50` |
| `IMAGE_DOWNLOAD_CONCURRENCY` | Parallel image downloads | `10` |
| `DEEPFAKE_ANALYSIS_CONCURRENCY` | Parallel Claude Vision analyses | `3` |

## License

MIT
