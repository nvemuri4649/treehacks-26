# GlazeGuard

**Native macOS Menu Bar App for Automatic Image Protection**

GlazeGuard is a macOS menu bar application that automatically detects when you copy images to your clipboard and applies adversarial perturbations (glazing) to protect against deepfake generation. It integrates seamlessly with your existing DiffusionGuard backend.

<p align="center">
  <img src="https://img.shields.io/badge/Platform-macOS%2013+-blue" alt="Platform">
  <img src="https://img.shields.io/badge/Swift-5.9-orange" alt="Swift">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

## Features

### 🛡️ Automatic Protection
- **Clipboard Monitoring**: Automatically detects when you copy images (Cmd+C)
- **Transparent Processing**: Glazes images behind the scenes with minimal user interaction
- **Smart Approval**: Shows consent dialog before processing (unless auto-approve is enabled)

### 🎭 Intelligent Masking
- **Face Detection**: Uses Apple Vision framework to automatically detect and protect faces
- **Full Image Mode**: Option to protect entire images
- **Mask Validation**: Ensures masks are suitable before processing

### 🎨 Beautiful UI
- **Translucent Overlay**: Native macOS vibrancy with animated progress display
- **Menu Bar Integration**: Lives in your menu bar, out of the way
- **Progress Tracking**: Real-time progress with time estimates

### ⚙️ Flexible Configuration
- **Multi-Backend Support**: Connect to local, GX10, or RunPod GPU servers
- **Glazing Strength**: Choose from Low (100), Medium (200), High (500), or Maximum (1000) iterations
- **Customizable Settings**: Full control over monitoring, notifications, and protection modes

## Screenshots

*(Coming soon)*

## Installation

### Prerequisites

- macOS 13.0 (Ventura) or later
- Xcode 15.0 or later
- DiffusionGuard backend server running (see [server setup](../server/README.md))

### Building from Source

1. **Clone the repository** (if not already done):
   ```bash
   cd /path/to/treehacks-26
   ```

2. **Open in Xcode**:
   ```bash
   cd GlazeGuard
   open Package.swift
   ```

3. **Build and Run**:
   - Select "GlazeGuard" scheme
   - Press Cmd+R to build and run
   - The app will appear in your menu bar as a shield icon

### Alternative: Build with Swift Package Manager

```bash
cd GlazeGuard
swift build -c release
# Executable will be at: .build/release/GlazeGuard
```

## Quick Start

### 1. Start the Backend Server

First, ensure your DiffusionGuard backend is running:

```bash
cd ../server
python app.py
```

The server should start on port 8888. You should see:
```
Starting DiffusionGuard + Fawkes API server on :8888
```

### 2. Configure Backend

GlazeGuard automatically reads `backends.json` from the project root. The default configuration includes:

- **local**: `http://localhost:8888` (for local testing)
- **gx10**: `http://spark-abcd.local:8888` (ASUS GX10 on local network)
- **runpod**: `https://{POD_ID}-8888.proxy.runpod.net` (RunPod GPU instance)

Edit `backends.json` to add your own backends or change the default.

### 3. Launch GlazeGuard

Run the app from Xcode or execute the built binary. The shield icon will appear in your menu bar.

### 4. Test Protection

**Method 1: Clipboard Monitoring**
1. Copy an image with a face (Cmd+C in any app)
2. GlazeGuard will detect it and show an approval dialog
3. Choose glazing strength and click "Protect Image"
4. Wait for the overlay to show progress (~2-3 minutes for 200 iterations)
5. Paste the image (Cmd+V) - it's now protected!

**Method 2: Drag and Drop**
1. Drag an image file onto the shield icon in the menu bar
2. Follow the approval dialog
3. The protected image will be saved with `_protected` suffix

**Method 3: File Picker**
1. Click the shield icon
2. Select "Protect Image..."
3. Choose a file and configure settings

## Architecture

### Project Structure

```
GlazeGuard/
├── App/
│   ├── GlazeGuardApp.swift          # Main app entry point
│   ├── AppState.swift               # Shared state management
│   └── MenuBarController.swift      # Menu bar UI and actions
├── Services/
│   ├── BackendService.swift         # HTTP client for /protect API
│   ├── PasteboardMonitor.swift      # Clipboard monitoring service
│   ├── MaskGenerator.swift          # Vision-based face detection
│   └── GlazingQueue.swift           # Job queue management
├── UI/
│   ├── OverlayWindow.swift          # Translucent system overlay
│   ├── GlazingProgressView.swift    # Animated progress view
│   ├── SettingsView.swift           # Configuration panel
│   └── ApprovalDialog.swift         # User consent dialog
└── Models/
    ├── BackendConfig.swift          # Parse backends.json
    ├── GlazingJob.swift             # Job state tracking
    └── Settings.swift               # User preferences
```

### Data Flow

```
1. Image Copied to Clipboard
   └─> PasteboardMonitor detects change
       └─> AppState.handleImageCopied()
           └─> Show ApprovalDialog (if not auto-approve)
               └─> User approves with settings
                   └─> Create GlazingJob
                       ├─> MaskGenerator creates face mask (Vision)
                       ├─> BackendService.protectImage() (HTTP POST)
                       └─> OverlayWindow shows progress
                           └─> Job completes
                               ├─> Replace clipboard with protected image
                               └─> Show notification
```

### Backend Integration

GlazeGuard communicates with the Flask backend via HTTP:

**Health Check**:
```
GET /health
Response: {"status": "ok", "gpu": {"name": "NVIDIA A100", "memory": "40GB"}}
```

**Image Protection**:
```
POST /protect?iters=200
Content-Type: multipart/form-data
Body: image=<png data>, mask=<png data>
Response: <protected png image>
```

See `BackendService.swift` for implementation details.

## Configuration

### Settings

Settings are stored in `~/Library/Application Support/GlazeGuard/config.json`:

```json
{
  "enabled": true,
  "backend": "local",
  "monitor_pasteboard": true,
  "auto_approve": false,
  "default_iterations": 200,
  "mask_mode": "auto_face",
  "show_notifications": true,
  "launch_at_login": false
}
```

### Backend Configuration

Edit `backends.json` in the project root to add custom backends:

```json
{
  "backends": {
    "my-server": {
      "url": "http://192.168.1.100:8888",
      "description": "My custom GPU server",
      "type": "local"
    }
  },
  "default": "my-server"
}
```

## Usage Tips

### Glazing Strengths

| Strength | Iterations | Time | Protection Level | Use Case |
|----------|-----------|------|------------------|----------|
| Low      | 100       | ~40s | Basic            | Quick protection |
| Medium   | 200       | ~1.5m | Good             | Recommended default |
| High     | 500       | ~3m  | Strong           | Important images |
| Maximum  | 1000      | ~6m  | Maximum          | Critical protection |

### Best Practices

1. **Use Face Detection** for photos with people - it's faster and focuses protection where it matters
2. **Use Full Image** for artwork, screenshots, or images without faces
3. **Enable Auto-Approve** if you trust the default settings and want seamless protection
4. **Monitor Backend Status** - the menu bar icon shows connection status

### Keyboard Shortcuts

- `Cmd+,` - Open Settings
- `Cmd+E` - Enable/Disable GlazeGuard
- `Cmd+M` - Start/Stop Clipboard Monitoring
- `Cmd+O` - Protect Image from File
- `Cmd+Q` - Quit

## Troubleshooting

### "Backend Unavailable" Error

**Problem**: Menu bar shows ❌ Backend: local (unavailable)

**Solutions**:
1. Check that server is running: `curl http://localhost:8888/health`
2. Verify `backends.json` has correct port (8888, not 5000)
3. Check firewall settings if using remote backend
4. Try switching to a different backend in Settings

### Images Not Being Detected

**Problem**: Clipboard monitoring doesn't trigger

**Solutions**:
1. Check that monitoring is enabled (menu bar or Settings)
2. Verify GlazeGuard is enabled (`Cmd+E`)
3. Try copying the image again
4. Check Console.app for debug logs (search "GlazeGuard")

### Glazing Takes Too Long

**Problem**: Protection process times out or is too slow

**Solutions**:
1. Reduce glazing strength (try Low or Medium)
2. Use Face Detection instead of Full Image
3. Check GPU utilization on backend server
4. Try a different backend with more powerful GPU

### Vision Framework Errors

**Problem**: Face detection fails

**Solutions**:
1. Ensure image is valid (not corrupted)
2. Try Full Image mode if face detection isn't critical
3. Check macOS version (requires macOS 13+)

## Development

### Running Tests

```bash
swift test
```

### Debug Mode

To see detailed logs, run from Xcode with Console.app open:

1. Open Console.app
2. Filter by "GlazeGuard"
3. Run app from Xcode (Cmd+R)
4. Look for logs prefixed with emoji icons:
   - 🚀 Startup
   - 📋 Clipboard events
   - 🔐 Protection operations
   - ✅ Success
   - ❌ Errors

### Adding Features

See `CONTRIBUTING.md` for guidelines on contributing new features.

## Roadmap

### Phase 1: MVP ✅ (Current)
- [x] Menu bar app
- [x] Clipboard monitoring
- [x] Face detection
- [x] Translucent overlay
- [x] Backend integration

### Phase 2: Auto-Detection (Planned)
- [ ] FSEvents monitoring for file system changes
- [ ] Detect browser downloads
- [ ] Watch ~/Desktop, ~/Downloads, ~/Pictures
- [ ] Automatic glazing of new files

### Phase 3: Advanced Features (Future)
- [ ] Network Extension for HTTP upload interception
- [ ] Batch processing multiple images
- [ ] Finder Quick Action ("Protect with GlazeGuard")
- [ ] Server-Sent Events for real-time progress
- [ ] Custom watermarks
- [ ] Integration with cloud storage (Dropbox, iCloud)

## FAQ

**Q: Does GlazeGuard modify my original images?**
A: For clipboard operations, yes - it replaces the clipboard with the protected version. For file operations, it saves a new file with `_protected` suffix, leaving the original intact.

**Q: Can I use GlazeGuard without an internet connection?**
A: Yes, if you run the backend server locally. The app only needs network access to reach the backend.

**Q: How much GPU memory is required?**
A: Depends on image size and model. Typically 4-8GB VRAM. See the main DiffusionGuard documentation for details.

**Q: Is GlazeGuard sandboxed?**
A: No, the app requires sandbox exemption to monitor the system clipboard. It only accesses the clipboard and user-selected files.

**Q: Can I run multiple backends simultaneously?**
A: Not currently. You can switch backends in Settings, but only one is active at a time.

**Q: Does this work with Retina displays?**
A: Yes, GlazeGuard handles high-DPI displays correctly.

## License

GlazeGuard is part of the DiffusionGuard project.
Copyright © 2026 DiffusionGuard Team. All rights reserved.

See [LICENSE](../LICENSE) for details.

## Credits

**Built with**:
- SwiftUI & AppKit
- Apple Vision Framework
- DiffusionGuard Backend

**Inspired by**:
- Glaze (University of Chicago)
- Fawkes Privacy Protection
- macOS Screen Time overlay design

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/diffusionguard/issues)
- **Documentation**: [Full Docs](../docs/)
- **Backend Setup**: [Server README](../server/README.md)

---

**Made with 🛡️ by the DiffusionGuard Team**
