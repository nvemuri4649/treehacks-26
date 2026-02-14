# Adversarial Agent Loop for Glazing Optimization

## Overview

The **Agent Loop** is an automated adversarial testing system that finds the optimal glazing strength to protect images against deepfakes. It uses Claude's vision capabilities to evaluate deepfake quality in an iterative loop.

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    ADVERSARIAL AGENT LOOP                       │
└─────────────────────────────────────────────────────────────────┘

Start with N=100 PGD iterations
         │
         ▼
    ┌────────────────┐
    │ 1. GLAZE       │  Apply DiffusionGuard with N iterations
    │    IMAGE       │  → Protected image
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │ 2. GENERATE    │  Run inpainting on protected image
    │    DEEPFAKE    │  → Deepfake output
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │ 3. RATE        │  Claude evaluates deepfake (1-10 scale)
    │    QUALITY     │  → Score + reasoning
    └────────┬───────┘
             │
             ▼
         Score > Threshold?
             │
       ┌─────┴─────┐
       │ YES       │ NO
       ▼           ▼
   Increase N   SUCCESS!
   (N += 100)   Found optimal
       │         glazing strength
       └─────┐
             │
             ▼
         (Repeat)
```

## Architecture

### Components

1. **Glazing Engine** (`server/app.py`)
   - Applies DiffusionGuard protection with configurable PGD iterations
   - Runs on GPU backend (GX10 / RunPod / SSH)

2. **Deepfake Generator** (`server/app.py`)
   - Uses Stable Diffusion Inpainting to generate deepfakes
   - Tests whether glazing successfully disrupted the model

3. **Rater Agent** (`client/rater_agent.py`)
   - Claude Opus 4.6 with vision capabilities
   - Evaluates deepfake quality on 1-10 scale
   - Provides detailed reasoning for ratings

4. **Loop Orchestrator** (`client/agent_loop.py`)
   - Coordinates the adversarial loop
   - Tracks iterations and saves results
   - Generates visual summaries

### Rating Scale

The Claude rater agent uses this scale:

| Score | Quality | Protection Status | Description |
|-------|---------|-------------------|-------------|
| 1-3 | Very Poor | **EXCELLENT** ✓ | Obvious artifacts, distortions, or failures. Glazing worked perfectly. |
| 4-6 | Mediocre | **GOOD** ✓ | Noticeable quality issues. Glazing is effective. |
| 7-8 | Good | **PARTIAL** ⚠️ | Mostly convincing but with subtle issues. Glazing partially worked. |
| 9-10 | Excellent | **FAILED** ✗ | Highly realistic. Glazing did not protect the image. |

## Setup

### 1. Install Client Dependencies

```bash
pip install -r client/requirements.txt
```

This installs:
- `anthropic` - Claude API SDK
- `requests` - HTTP client
- `pillow` - Image processing

### 2. Set Up Anthropic API Key

Get your API key from [console.anthropic.com](https://console.anthropic.com/):

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

Or pass directly:
```bash
python client/agent_loop.py --anthropic-api-key "sk-ant-api03-..." ...
```

### 3. Ensure GPU Backend is Running

The agent loop needs a running DiffusionGuard server. See [README.md](README.md) for setup.

Quick check:
```bash
# Check GX10
curl http://spark-abcd.local:5000/health

# Check RunPod
curl https://your-pod-5000.proxy.runpod.net/health
```

## Usage

### Basic Usage

```bash
python client/agent_loop.py \
    --image face.png \
    --mask mask.png \
    --prompt "a person in jail" \
    --threshold 6
```

This will:
1. Start with 100 PGD iterations
2. Generate deepfakes and rate them with Claude
3. Increase iterations by 100 each loop
4. Stop when deepfake score ≤ 6/10 (glazing is effective)

### Advanced Options

```bash
python client/agent_loop.py \
    --image photosofnikhil/nikhil.png \
    --mask photosofnikhil/nikhil_mask.png \
    --prompt "a person being arrested" \
    --threshold 5 \
    --start-iters 50 \
    --iter-increment 100 \
    --max-iters 1000 \
    --max-iterations 10 \
    --backend gx10 \
    --output-dir results_nikhil
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--image` | Required | Path to original image |
| `--mask` | Required | Path to mask (white = protected region) |
| `--prompt` | Required | Inpainting prompt for deepfake generation |
| `--threshold` | 6 | Target score (1-10). Loop until score ≤ threshold |
| `--start-iters` | 100 | Starting PGD iterations |
| `--iter-increment` | 100 | How much to increase each loop |
| `--max-iters` | 1000 | Maximum PGD iterations to try |
| `--max-iterations` | 10 | Maximum loop iterations |
| `--backend` | (default) | GPU backend: `gx10`, `runpod`, `local` |
| `--output-dir` | Auto | Output directory (default: `agent_loop_results_TIMESTAMP`) |
| `--seed` | 42 | Random seed for reproducibility |

## Output Structure

The agent loop creates a timestamped directory with comprehensive results:

```
agent_loop_results_20260214_123456/
├── original.png                          # Original input image
├── iter01_protected_100iters.png         # Protected image (iteration 1, 100 PGD iters)
├── iter01_deepfake_100iters.png          # Deepfake from protected image
├── iter01_summary.png                    # Visual summary with Claude's rating
├── iter02_protected_200iters.png         # Iteration 2 (200 PGD iters)
├── iter02_deepfake_200iters.png
├── iter02_summary.png
├── ...
└── results.json                           # Complete results with all metrics
```

### Summary Images

Each iteration generates a summary image showing:
- Original image (reference)
- Protected image (with N PGD iterations)
- Deepfake generated from protected image
- Claude's score (1-10) and reasoning
- Protection effectiveness indicator (EFFECTIVE / PARTIAL / WEAK)

### Results JSON

```json
{
  "config": {
    "image": "face.png",
    "mask": "mask.png",
    "prompt": "a person in jail",
    "threshold": 6,
    "start_iters": 100,
    "iter_increment": 100,
    "max_iters": 1000
  },
  "results": [
    {
      "iteration": 1,
      "pgd_iters": 100,
      "score": 8,
      "reasoning": "The deepfake is mostly convincing with subtle artifacts...",
      "glaze_time": 45.2,
      "deepfake_time": 12.3,
      "protected_path": "iter01_protected_100iters.png",
      "deepfake_path": "iter01_deepfake_100iters.png",
      "summary_path": "iter01_summary.png"
    },
    {
      "iteration": 2,
      "pgd_iters": 200,
      "score": 5,
      "reasoning": "Noticeable quality degradation with visible artifacts...",
      "glaze_time": 89.7,
      "deepfake_time": 12.1
    }
  ],
  "success": true,
  "final_score": 5,
  "optimal_iters": 200,
  "total_iterations": 2
}
```

## Examples

### Example 1: Find Optimal Glazing for Face Protection

```bash
python client/agent_loop.py \
    --image photosofnikhil/nikhil.png \
    --mask photosofnikhil/nikhil_mask.png \
    --prompt "a person in a courtroom trial" \
    --threshold 6 \
    --backend gx10
```

**Expected Output:**
```
================================================================================
ADVERSARIAL GLAZING AGENT LOOP
================================================================================
  Target threshold: 6/10
  Starting PGD iterations: 100
  Iteration increment: +100
  ...

================================================================================
ITERATION 1 — Testing with 100 PGD iterations
================================================================================
[1/3] Glazing image (100 iterations)...
      ✓ Complete in 45.3s
[2/3] Generating deepfake...
      ✓ Complete in 12.1s
[3/3] Rating deepfake quality with Claude...
      ✓ Score: 8/10
      Reasoning: The deepfake shows good quality with only minor artifacts...

Score 8/10 > threshold 6/10. Increasing to 200 iterations...

================================================================================
ITERATION 2 — Testing with 200 PGD iterations
================================================================================
[1/3] Glazing image (200 iterations)...
      ✓ Complete in 91.2s
[2/3] Generating deepfake...
      ✓ Complete in 12.3s
[3/3] Rating deepfake quality with Claude...
      ✓ Score: 5/10
      Reasoning: Significant quality degradation with visible distortions...

================================================================================
🎉 SUCCESS! Glazing is effective!
================================================================================
  Final score: 5/10 (threshold: 6/10)
  Optimal PGD iterations: 200
  Deepfake quality is now below threshold.
  Protection is working effectively!
```

### Example 2: Test Aggressive Glazing

```bash
python client/agent_loop.py \
    --image face.png \
    --mask mask.png \
    --prompt "a person as a superhero" \
    --threshold 4 \
    --start-iters 200 \
    --iter-increment 200 \
    --max-iters 1200 \
    --backend runpod
```

This tests stronger protection by:
- Setting a stricter threshold (4 instead of 6)
- Starting at 200 iterations instead of 100
- Increasing by 200 each loop instead of 100

### Example 3: Quick Test with Local Backend

```bash
python client/agent_loop.py \
    --image test.png \
    --mask test_mask.png \
    --prompt "a robot" \
    --threshold 7 \
    --max-iterations 3 \
    --backend local
```

This runs a quick test (max 3 iterations) on your local GPU for rapid prototyping.

## Testing the Rater Agent Standalone

You can test the Claude rater agent independently:

```bash
python client/rater_agent.py \
    --original test_results/original.png \
    --deepfake test_results/inpaint_protected.png \
    --mask test_images/mask.png \
    --context "a person in a hospital"
```

**Output:**
```
======================================================================
DEEPFAKE QUALITY SCORE: 4/10
======================================================================
The deepfake shows significant quality degradation with noticeable
artifacts in the facial region. The lighting and textures are
inconsistent, indicating that the protection successfully disrupted
the generation process.
======================================================================
```

## How Claude Evaluates Deepfakes

The rater agent uses Claude Opus 4.6's vision capabilities with a specialized prompt:

1. **Input**: Original image, deepfake, and mask
2. **Analysis**: Claude examines:
   - Photorealism and natural appearance
   - Consistency of lighting, colors, textures
   - Artifacts, distortions, unnatural elements
   - How well modifications blend with the rest of the image
3. **Output**: Structured JSON with:
   - Score (1-10)
   - Detailed reasoning
   - Specific artifacts observed
   - Protection effectiveness assessment

The agent is calibrated to be a tough evaluator, representing a knowledgeable adversary trying to create convincing deepfakes.

## Troubleshooting

### API Key Issues

```
ERROR: Anthropic API key required.
```

**Solution**: Set the environment variable:
```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

### Server Connection Issues

```
ERROR: Cannot reach server at http://...
```

**Solution**:
1. Check the server is running: `curl http://server:5000/health`
2. Verify backend configuration in `backends.json`
3. For RunPod, ensure pod is running and port 5000 is exposed

### GPU Out of Memory

If glazing fails with OOM errors, reduce batch size or use a smaller model variant on the server.

### Claude API Rate Limits

If you hit rate limits, the loop will fail. Solutions:
- Add delays between iterations (modify `agent_loop.py`)
- Use a lower-tier Claude model (change `model="claude-sonnet-4.5"` in `rater_agent.py`)
- Reduce `--max-iterations`

## Performance Notes

### Timing Expectations

On NVIDIA DGX Spark (GX10):
- **Glazing**: ~0.4s per PGD iteration (e.g., 200 iters ≈ 90s)
- **Deepfake Generation**: ~10-15s (50 diffusion steps)
- **Claude Rating**: ~2-5s per image

Total per iteration: ~2-3 minutes for 200 PGD iterations

### Cost Estimates

Claude API costs (as of Feb 2025):
- Claude Opus 4.6: ~$0.03 per image rating (3 images × vision tokens)
- Average loop (5 iterations): ~$0.15 total

GPU costs:
- Local (GX10): Free if you own hardware
- RunPod: ~$0.50-1.00/hour depending on GPU

## Integration with DiffusionGuard Pipeline

The agent loop integrates seamlessly with the existing DiffusionGuard pipeline:

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Client    │─────▶│ GPU Backend  │◀────▶│   Claude    │
│ agent_loop  │      │ DiffusionGrd │      │    Rater    │
└─────────────┘      └──────────────┘      └─────────────┘
      │                      │                      │
      │ /protect (glaze)     │                      │
      ├─────────────────────▶│                      │
      │                      │                      │
      │ /test-inpaint        │                      │
      ├─────────────────────▶│                      │
      │                      │                      │
      │                      │ rate_deepfake()      │
      ├──────────────────────┴─────────────────────▶│
      │                                              │
      │◀─────────────────────────────────────────────┤
      │ score + reasoning                            │
      │                                              │
      ▼
   Results & Summary
```

## Future Enhancements

Possible improvements to the agent loop:

1. **Multi-Agent Consensus**: Use multiple Claude instances for more robust ratings
2. **Perceptual Metrics**: Incorporate LPIPS, SSIM, or FID scores alongside Claude ratings
3. **Dynamic Scheduling**: Intelligently adjust iteration increments based on score changes
4. **Parallel Testing**: Test multiple iteration values simultaneously to find optimal faster
5. **A/B Comparison**: Have Claude compare pairs side-by-side for more reliable ratings
6. **Video Support**: Extend to video deepfakes using Stable Video Diffusion
7. **Face Recognition Attack**: Test against face recognition models (Fawkes-style)

## References

- [DiffusionGuard Paper (ICLR 2025)](https://arxiv.org/abs/2410.05694)
- [Anthropic Claude API](https://docs.anthropic.com/claude/reference/messages)
- [Stable Diffusion Inpainting](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint)

## Citation

If you use this agent loop system in research, please cite:

```bibtex
@inproceedings{choi2025diffusionguard,
  title={DiffusionGuard: A Robust Defense Against Diffusion-Based Image Manipulation},
  author={Choi, et al.},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```
