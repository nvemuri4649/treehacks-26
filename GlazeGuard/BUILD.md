# Building GlazeGuard

Complete build instructions for the GlazeGuard macOS application.

## Prerequisites

### System Requirements
- macOS 13.0 (Ventura) or later
- Xcode 15.0 or later
- Swift 5.9 or later
- Command Line Tools: `xcode-select --install`

### Backend Requirements
- Python 3.10+
- DiffusionGuard backend server (see `../server/`)
- GPU recommended (NVIDIA with CUDA support)

## Build Methods

### Method 1: Xcode (Recommended)

1. **Open Project**:
   ```bash
   cd GlazeGuard
   open Package.swift
   ```

2. **Configure Signing** (Xcode):
   - Select "GlazeGuard" target
   - Go to "Signing & Capabilities"
   - Select your development team
   - Xcode will automatically manage provisioning

3. **Build**:
   - Select scheme: "GlazeGuard" → "My Mac"
   - Press `Cmd+B` to build
   - Press `Cmd+R` to run

4. **Find Binary**:
   ```bash
   # Debug build
   ~/Library/Developer/Xcode/DerivedData/GlazeGuard-*/Build/Products/Debug/GlazeGuard.app

   # Release build (select "Edit Scheme" → Run → Release)
   ~/Library/Developer/Xcode/DerivedData/GlazeGuard-*/Build/Products/Release/GlazeGuard.app
   ```

### Method 2: Swift Package Manager

1. **Build Release**:
   ```bash
   cd GlazeGuard
   swift build -c release
   ```

2. **Find Binary**:
   ```bash
   .build/release/GlazeGuard
   ```

3. **Create App Bundle** (optional):
   ```bash
   ./scripts/create_app_bundle.sh
   ```

### Method 3: Command Line with Xcode

1. **Build**:
   ```bash
   cd GlazeGuard
   xcodebuild -scheme GlazeGuard -configuration Release -derivedDataPath build
   ```

2. **Find App**:
   ```bash
   build/Build/Products/Release/GlazeGuard.app
   ```

## Configuration

### 1. Backend Setup

Before running GlazeGuard, ensure the backend server is configured:

```bash
cd ../server
python app.py
```

Verify it's running:
```bash
curl http://localhost:8888/health
```

Expected response:
```json
{
  "status": "ok",
  "gpu": {
    "name": "NVIDIA RTX 4090",
    "memory": "24GB"
  }
}
```

### 2. Backend Configuration File

Edit `../backends.json` to configure your backends:

```json
{
  "backends": {
    "local": {
      "url": "http://localhost:8888",
      "description": "Local machine",
      "type": "local"
    },
    "gx10": {
      "url": "http://spark-abcd.local:8888",
      "description": "ASUS GX10 GPU Server",
      "type": "ssh-docker",
      "ssh": "user@spark-abcd.local"
    }
  },
  "default": "local"
}
```

**Important**: Ensure all URLs use port **8888** (not 5000).

### 3. App Configuration

First launch will create default settings:
```bash
~/Library/Application Support/GlazeGuard/config.json
```

You can edit this manually or use the Settings UI (Cmd+,).

## Building for Distribution

### Create Release Build

1. **Archive in Xcode**:
   - Product → Archive
   - Organizer window opens
   - Select archive → Distribute App
   - Choose distribution method:
     - "Developer ID" for distribution outside Mac App Store
     - "Mac App Store" for App Store submission

2. **Notarization** (for distribution):
   ```bash
   # Submit for notarization
   xcrun notarytool submit GlazeGuard.zip \
     --apple-id "your@email.com" \
     --team-id "TEAM_ID" \
     --password "app-specific-password"

   # Wait for approval, then staple
   xcrun stapler staple GlazeGuard.app
   ```

### Create DMG Installer

```bash
# Create DMG for distribution
hdiutil create -volname "GlazeGuard" \
  -srcfolder GlazeGuard.app \
  -ov -format UDZO \
  GlazeGuard-1.0.0.dmg
```

## Troubleshooting

### Build Errors

**Error**: "No such module 'Vision'"

**Solution**: Ensure deployment target is macOS 13.0+:
```swift
// Package.swift
platforms: [
    .macOS(.v13)
]
```

---

**Error**: "Code signing failed"

**Solutions**:
1. Select your development team in Xcode
2. Or disable signing for debug builds:
   ```bash
   xcodebuild CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO
   ```

---

**Error**: "Cannot find 'backends.json'"

**Solution**: App looks for backends.json in these locations:
1. `/Users/shahabhishek1729/Documents/Stanford/Notes/EC/TreeHacks26/treehacks-26/backends.json` (hardcoded during development)
2. App bundle resources

For distribution, copy backends.json to Resources:
```bash
cp ../backends.json GlazeGuard/Resources/
```

### Runtime Issues

**Issue**: "Backend unavailable" on launch

**Checks**:
1. Is server running? `curl http://localhost:8888/health`
2. Is port correct in backends.json? (should be 8888)
3. Is firewall blocking connections?
4. Check Console.app logs for "GlazeGuard"

---

**Issue**: Face detection not working

**Checks**:
1. macOS version >= 13.0?
2. Is app sandboxed? (should be disabled for Vision framework)
3. Check image format (PNG, JPEG supported)

## Development Build

For active development with faster iteration:

```bash
# Build in debug mode (includes symbols)
swift build

# Run with debug output
.build/debug/GlazeGuard
```

Enable debug logging in Xcode:
- Edit Scheme → Run → Arguments
- Add environment variable: `DEBUG=1`

## Performance Optimization

### Release Builds

Swift compiler optimizations are enabled automatically in release builds:
- `-O` (optimize for speed)
- `-whole-module-optimization`
- Strip debug symbols

### Reduce Build Time

1. **Use Parallel Building**:
   ```bash
   swift build -j$(sysctl -n hw.ncpu)
   ```

2. **Incremental Builds**:
   - Only changed files are recompiled
   - Clean rarely: `swift package clean`

3. **Precompiled Modules**:
   - Frameworks are cached after first build

## Testing

### Run Unit Tests
```bash
swift test
```

### Manual Testing Checklist

- [ ] App appears in menu bar
- [ ] Health check shows backend status
- [ ] Copy image triggers detection
- [ ] Approval dialog appears
- [ ] Overlay shows progress
- [ ] Protected image appears in pasteboard
- [ ] Drag-drop to menu bar works
- [ ] Settings persist between launches
- [ ] Notifications appear
- [ ] Backend switching works

### End-to-End Test

```bash
# Terminal 1: Start backend
cd ../server
python app.py

# Terminal 2: Build and run app
cd GlazeGuard
swift build && .build/debug/GlazeGuard
```

Then:
1. Copy an image with a face
2. Approve protection (Medium strength)
3. Wait for overlay to complete
4. Paste image and verify it's different

## Continuous Integration

### GitHub Actions Example

```yaml
name: Build GlazeGuard

on: [push, pull_request]

jobs:
  build:
    runs-on: macos-13
    steps:
      - uses: actions/checkout@v3
      - name: Build
        run: |
          cd GlazeGuard
          swift build -c release
      - name: Test
        run: |
          cd GlazeGuard
          swift test
```

## Clean Build

If you encounter strange issues, clean everything:

```bash
# Swift PM
swift package clean
rm -rf .build

# Xcode derived data
rm -rf ~/Library/Developer/Xcode/DerivedData/GlazeGuard-*

# App settings (reset configuration)
rm -rf ~/Library/Application\ Support/GlazeGuard
```

## Additional Resources

- [Swift Package Manager](https://www.swift.org/package-manager/)
- [Xcode Build Settings](https://developer.apple.com/documentation/xcode/build-settings-reference)
- [Code Signing Guide](https://developer.apple.com/support/code-signing/)
- [App Notarization](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)

---

**Questions?** Open an issue or check the main [README.md](README.md).
