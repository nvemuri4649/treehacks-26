# GlazeGuard Quick Start

Get GlazeGuard running in under 5 minutes.

## Prerequisites Check

```bash
# Check macOS version (need 13.0+)
sw_vers

# Check Swift
swift --version  # Need 5.9+

# Check Xcode
xcodebuild -version  # Need 15.0+
```

## Step 1: Start the Backend Server

Open a terminal and start the DiffusionGuard backend:

```bash
cd ../server
python app.py
```

**Expected output**:
```
Loading Stable Diffusion model...
Starting DiffusionGuard + Fawkes API server on :8888
```

**Verify it's running**:
```bash
curl http://localhost:8888/health
```

Should return:
```json
{
  "status": "ok",
  "gpu": {"name": "...", "memory": "..."}
}
```

✅ If you see this, your backend is ready!

## Step 2: Build GlazeGuard

Choose your preferred method:

### Option A: Using Make (Easiest)

```bash
cd GlazeGuard
make run
```

That's it! The app will build and launch automatically.

### Option B: Using Xcode (Best for Development)

```bash
cd GlazeGuard
open Package.swift
```

Then press `Cmd+R` to build and run.

### Option C: Using Swift Directly

```bash
cd GlazeGuard
swift build
.build/debug/GlazeGuard
```

## Step 3: Test It Works

1. **Check the menu bar** - you should see a shield icon (🛡️)

2. **Click the shield icon** - menu should show:
   ```
   GlazeGuard
   ✅ Backend: local
   Disable GlazeGuard
   Stop Monitoring Clipboard
   Protect Image...
   Settings...
   Quit GlazeGuard
   ```

3. **Test protection**:
   - Find an image with a face
   - Copy it (Cmd+C)
   - GlazeGuard should show a dialog: "Protect This Image?"
   - Click "Protect Image"
   - Wait for the overlay (~1-2 minutes for 200 iterations)
   - Paste the image (Cmd+V) - it should be protected!

## Verify Protection Worked

The protected image should:
- Look nearly identical to the original (subtle perturbations)
- Have slight noise when zoomed in
- Prevent deepfake generation when uploaded

Compare original vs protected:
```bash
# Save both versions and compare file sizes
# Protected version should be similar size but slightly different
```

## Common Issues

### ❌ "Backend unavailable"

**Check**:
```bash
# Is the server running?
curl http://localhost:8888/health

# Is the port correct in backends.json?
cat ../backends.json | grep url
# Should show: "url": "http://localhost:8888"  (NOT 5000)
```

**Fix**:
- Make sure server is running (Step 1)
- Check backends.json uses port 8888

### ❌ "No faces detected, using full-image mask"

**This is normal!** If the image has no faces, GlazeGuard will protect the entire image instead.

To test face detection:
- Use an image with clear, frontal faces
- Try a photo from your photo library

### ❌ App doesn't appear in menu bar

**Check**:
1. Did the build succeed? Look for "Build succeeded" in terminal
2. Is the app actually running? Check Activity Monitor for "GlazeGuard"
3. Try restarting the app

### ❌ "Image copied to pasteboard" but no dialog appears

**Check Settings**:
```bash
# Check if monitoring is enabled
# Click shield icon → should say "Stop Monitoring Clipboard" (not "Start")
```

**Try**:
- Click shield → "Start Monitoring Clipboard"
- Or press `Cmd+M` to toggle monitoring

## Next Steps

### Configure Settings

Click shield icon → Settings (or press `Cmd+,`):

1. **Choose glazing strength**:
   - Low (100 iters) - ~40s, basic protection
   - Medium (200 iters) - ~1.5m, recommended ✅
   - High (500 iters) - ~3m, strong protection
   - Maximum (1000 iters) - ~6m, maximum protection

2. **Choose protection mode**:
   - Auto-detect Faces - faster, focused on faces ✅
   - Full Image - protects entire image

3. **Enable auto-approve** (optional):
   - Skip the approval dialog for automatic protection
   - Be careful: all copied images will be glazed!

### Use Different Backends

If you have access to a GPU server:

1. **Edit backends.json**:
   ```json
   {
     "backends": {
       "local": { ... },
       "my-gpu": {
         "url": "http://192.168.1.100:8888",
         "description": "My GPU Server",
         "type": "local"
       }
     },
     "default": "my-gpu"
   }
   ```

2. **Switch backend**:
   - Click shield → Settings
   - Backend dropdown → select "my-gpu"
   - Click Save

### Try Drag-and-Drop

Instead of clipboard:
1. Drag an image file from Finder
2. Drop it on the shield icon in menu bar
3. Choose settings and protect

### Protect from Files

Click shield → "Protect Image...":
1. Choose an image file
2. Configure protection
3. Protected image saved as `<original>_protected.png`

## Full Workflow Example

**Scenario**: Protect profile picture before uploading to social media

```bash
# 1. Start backend (Terminal 1)
cd server && python app.py

# 2. Build and run GlazeGuard (Terminal 2)
cd GlazeGuard && make run

# 3. In your photo app:
#    - Open profile photo
#    - Press Cmd+C to copy

# 4. In GlazeGuard dialog:
#    - Select "Medium (200)" strength
#    - Keep "Faces" mode
#    - Click "Protect Image"

# 5. Wait for overlay to complete (~1-2 min)

# 6. In social media:
#    - Paste (Cmd+V) in upload field
#    - The glazed version will be uploaded

✅ Your photo is now protected from deepfakes!
```

## Performance Tips

- **Use Face Detection** when possible (faster than full image)
- **Start with Low/Medium** strength - test if protection is adequate
- **Use GPU backend** for faster processing (local CPU is slower)
- **Close other GPU apps** to free up VRAM

## Keyboard Shortcuts

- `Cmd+,` - Settings
- `Cmd+E` - Enable/Disable
- `Cmd+M` - Toggle clipboard monitoring
- `Cmd+O` - Protect file
- `Cmd+Q` - Quit

## Need Help?

- **Logs**: Check Console.app, filter by "GlazeGuard"
- **Issues**: See [GitHub Issues](https://github.com/...)
- **Docs**: Full documentation in [README.md](README.md)
- **Build**: Detailed build instructions in [BUILD.md](BUILD.md)

---

**Congratulations!** 🎉 You're now protecting your images from deepfakes.
