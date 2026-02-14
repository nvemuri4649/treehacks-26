# Agent Loop Quickstart Guide

## What is the Agent Loop?

The agent loop is an **automated adversarial testing system** that finds the optimal glazing strength to protect your images from deepfakes. It uses Claude's vision AI to evaluate deepfake quality in a loop:

```
Start → Glaze → Generate Deepfake → Rate with Claude → Score too high? → Increase glazing → Repeat
                                                    ↓
                                              Score low enough!
                                                    ↓
                                            Glazing is effective ✓
```

## 5-Minute Setup

### Step 1: Install Dependencies

```bash
pip install -r client/requirements.txt
```

This installs:
- `anthropic` - Claude SDK
- `requests` - HTTP client
- `pillow` - Image processing

### Step 2: Get Anthropic API Key

1. Go to [console.anthropic.com](https://console.anthropic.com/)
2. Sign up or log in
3. Create an API key
4. Export it:

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

### Step 3: Verify Setup

```bash
python client/check_setup.py --backend gx10
```

This checks:
- ✓ Python dependencies installed
- ✓ API key is set
- ✓ Backend configuration
- ✓ GPU server is reachable
- ✓ Claude API is working

### Step 4: Run Your First Agent Loop

```bash
python client/agent_loop.py \
    --image photosofnikhil/nikhil.png \
    --mask photosofnikhil/nikhil_mask.png \
    --prompt "a person in jail" \
    --threshold 6 \
    --backend gx10
```

Or use the example script:

```bash
./example_agent_loop.sh
```

## What Happens During a Run?

### Iteration 1

```
[1/3] Glazing image (100 iterations)...
      ✓ Complete in 45.3s
[2/3] Generating deepfake...
      ✓ Complete in 12.1s
[3/3] Rating deepfake quality with Claude...
      ✓ Score: 8/10
      Reasoning: The deepfake is mostly convincing with minor artifacts...
```

Score 8 is too high (threshold is 6), so we increase glazing strength.

### Iteration 2

```
[1/3] Glazing image (200 iterations)...
      ✓ Complete in 91.2s
[2/3] Generating deepfake...
      ✓ Complete in 12.3s
[3/3] Rating deepfake quality with Claude...
      ✓ Score: 5/10
      Reasoning: Significant quality degradation, visible distortions...
```

Score 5 is below threshold! Success!

### Final Output

```
🎉 SUCCESS! Glazing is effective!
  Final score: 5/10 (threshold: 6/10)
  Optimal PGD iterations: 200
  Protection is working effectively!
```

## Understanding the Results

The loop creates a timestamped directory:

```
agent_loop_results_20260214_123456/
├── original.png                      # Your input image
├── iter01_protected_100iters.png     # Protected with 100 PGD iterations
├── iter01_deepfake_100iters.png      # Deepfake from protected image
├── iter01_summary.png                # Visual summary with Claude's rating
├── iter02_protected_200iters.png     # Protected with 200 PGD iterations
├── iter02_deepfake_200iters.png
├── iter02_summary.png
└── results.json                       # Complete results data
```

### Summary Images

Each `summary.png` shows:

```
┌─────────────────────────────────────────────────────────────────┐
│ Iteration 2 — 200 PGD Iterations — Score: 5/10                  │
├─────────────────────────────────────────────────────────────────┤
│  Original         Protected (200 iters)   Deepfake (Score: 5/10)│
│  ┌─────────┐     ┌─────────┐            ┌─────────┐            │
│  │         │     │         │            │         │            │
│  │  [IMG]  │     │  [IMG]  │            │  [IMG]  │            │
│  │         │     │         │            │         │            │
│  └─────────┘     └─────────┘            └─────────┘            │
│                                                                  │
│ Claude's Assessment:                                             │
│ Significant quality degradation with visible artifacts in the   │
│ facial region. Protection is working effectively.               │
│                                                                  │
│ Protection: EFFECTIVE ✓                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Rating Scale

| Score | Quality | Protection | What It Means |
|-------|---------|------------|---------------|
| 1-3 | Very Poor | **EXCELLENT** ✓ | Deepfake is broken, glazing worked perfectly |
| 4-6 | Mediocre | **GOOD** ✓ | Noticeable artifacts, glazing is effective |
| 7-8 | Good | **PARTIAL** ⚠️ | Mostly convincing, glazing partially worked |
| 9-10 | Excellent | **FAILED** ✗ | Highly realistic, glazing didn't protect |

## Common Use Cases

### Test Different Prompts

See how well glazing protects against different attack scenarios:

```bash
# Jail scenario
python client/agent_loop.py \
    --image face.png --mask mask.png \
    --prompt "a person in jail" --threshold 6

# Medical scenario
python client/agent_loop.py \
    --image face.png --mask mask.png \
    --prompt "a person in a hospital bed" --threshold 6

# Celebrity deepfake
python client/agent_loop.py \
    --image face.png --mask mask.png \
    --prompt "a person at an awards ceremony" --threshold 6
```

### Find Minimal Protection

Use a higher threshold to find the minimum glazing needed:

```bash
python client/agent_loop.py \
    --image face.png --mask mask.png \
    --prompt "a person in jail" \
    --threshold 7 \
    --start-iters 50 \
    --iter-increment 50
```

### Maximum Protection

Use a strict threshold for high-security scenarios:

```bash
python client/agent_loop.py \
    --image face.png --mask mask.png \
    --prompt "a person in jail" \
    --threshold 4 \
    --start-iters 200 \
    --max-iters 1200
```

## Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--threshold` | 6 | Stop when deepfake score ≤ this (1-10) |
| `--start-iters` | 100 | Initial PGD iterations to try |
| `--iter-increment` | 100 | How much to increase each loop |
| `--max-iters` | 1000 | Don't exceed this many PGD iterations |
| `--max-iterations` | 10 | Maximum loop iterations |

### Choosing a Threshold

- **Threshold 4-5**: Very strict, maximum protection, longer runtime
- **Threshold 6-7**: Balanced, good protection with reasonable runtime
- **Threshold 8-9**: Minimal protection, fast testing

### Iteration Strategy

**Fast (for testing):**
```bash
--start-iters 50 --iter-increment 50 --max-iters 400 --max-iterations 5
```

**Balanced (recommended):**
```bash
--start-iters 100 --iter-increment 100 --max-iters 800 --max-iterations 8
```

**Thorough (production):**
```bash
--start-iters 100 --iter-increment 50 --max-iters 1200 --max-iterations 15
```

## Troubleshooting

### "ANTHROPIC_API_KEY not set"

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

Get your key from [console.anthropic.com](https://console.anthropic.com/)

### "Cannot reach server"

Make sure your GPU backend is running:

```bash
# GX10
curl http://spark-abcd.local:5000/health

# RunPod
curl https://your-pod-5000.proxy.runpod.net/health

# Local
curl http://localhost:5000/health
```

Start the server if needed:
```bash
# GX10
ssh nikhil@spark-abcd.local
docker exec -it diffguard bash -c 'cd /workspace/project/server && python app.py'

# Local
python server/app.py
```

### "Could not parse structured response"

This is usually harmless - the script extracts the score from Claude's text response even if JSON parsing fails. The loop continues normally.

### Loop Not Converging

If the loop reaches `max-iters` without success:

1. **Increase max-iters**: Try `--max-iters 1500`
2. **Lower threshold**: Use `--threshold 7` instead of 6
3. **Check if glazing is working**: Look at the summary images - are they getting worse each iteration?

## Cost Estimates

**Claude API** (Feb 2025 pricing):
- ~$0.03 per image rating (3 images with vision)
- Average loop (5 iterations): ~$0.15

**GPU Usage**:
- Local (GX10): Free if you own it
- RunPod: ~$0.50-1.00/hour

**Typical loop runtime**: 10-20 minutes depending on iterations needed

## Next Steps

1. **Read the full docs**: See [AGENT_LOOP.md](AGENT_LOOP.md) for detailed information
2. **Test the rater standalone**: Try `python client/rater_agent.py --help`
3. **Experiment with parameters**: Adjust thresholds and iteration schedules
4. **Integrate with your workflow**: Use results to set optimal glazing for production

## Questions?

- Full documentation: [AGENT_LOOP.md](AGENT_LOOP.md)
- Main project README: [README.md](README.md)
- DiffusionGuard paper: [arxiv.org/abs/2410.05694](https://arxiv.org/abs/2410.05694)

## Example Session

Here's what a complete run looks like:

```
$ python client/agent_loop.py \
    --image face.png \
    --mask mask.png \
    --prompt "a person in jail" \
    --threshold 6

Connecting to GPU backend...
  Backend: 'gx10' http://spark-abcd.local:5000
  Server OK — GPU: NVIDIA A100-SXM4-80GB, Memory: 81.9 GB

================================================================================
ADVERSARIAL GLAZING AGENT LOOP
================================================================================
  Target threshold: 6/10
  Starting PGD iterations: 100
  Iteration increment: +100
  Max PGD iterations: 1000
  Max loop iterations: 10
  Deepfake prompt: "a person in jail"
  Output directory: agent_loop_results_20260214_143521
================================================================================

================================================================================
ITERATION 1 — Testing with 100 PGD iterations
================================================================================
[1/3] Glazing image (100 iterations)...
      ✓ Complete in 43.2s → iter01_protected_100iters.png
[2/3] Generating deepfake...
      ✓ Complete in 11.8s → iter01_deepfake_100iters.png
[3/3] Rating deepfake quality with Claude...
      ✓ Score: 8/10
      Reasoning: The deepfake shows good quality with only subtle artifacts...
      ✓ Summary saved → iter01_summary.png

Score 8/10 > threshold 6/10. Increasing to 200 iterations...

================================================================================
ITERATION 2 — Testing with 200 PGD iterations
================================================================================
[1/3] Glazing image (200 iterations)...
      ✓ Complete in 87.5s → iter02_protected_200iters.png
[2/3] Generating deepfake...
      ✓ Complete in 12.1s → iter02_deepfake_200iters.png
[3/3] Rating deepfake quality with Claude...
      ✓ Score: 5/10
      Reasoning: Noticeable quality degradation with visible distortions...
      ✓ Summary saved → iter02_summary.png

================================================================================
🎉 SUCCESS! Glazing is effective!
================================================================================
  Final score: 5/10 (threshold: 6/10)
  Optimal PGD iterations: 200
  Deepfake quality is now below threshold.
  Protection is working effectively!

================================================================================
SUMMARY
================================================================================
  Total iterations run: 2
  Results saved to: agent_loop_results_20260214_143521/results.json

Iteration breakdown:
  [1] 100 iters → Score 8/10 ✗ WEAK
  [2] 200 iters → Score 5/10 ✓ EFFECTIVE
================================================================================
```

Now open the output directory to see all the generated images and the full results!
