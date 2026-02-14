# GlazeGuard — macOS Menu Bar Image Protection

Native macOS menu bar app that automatically glazes images copied to clipboard, protecting against deepfake generation. Shows a translucent overlay with progress while DiffusionGuard runs on the GPU backend.

## How It Works

1. Copy an image (Cmd+C) — GlazeGuard detects it via clipboard monitoring
2. Approval dialog appears (or auto-approve if configured)
3. Translucent overlay shows glazing progress
4. Clipboard is replaced with the protected image — paste as normal

Also supports drag-and-drop onto the menu bar icon, or "Protect Image..." from the menu.

## Build & Run

```bash
cd GlazeGuard
swift build -c release          # or open Package.swift in Xcode, Cmd+R
```

Requires macOS 13+ and a running DiffusionGuard backend (see root README).

## Structure

```
App/           — Entry point, AppState, MenuBarController
Services/      — BackendService (HTTP), PasteboardMonitor, MaskGenerator (Vision), GlazingQueue
UI/            — OverlayWindow (translucent HUD), GlazingProgressView, ApprovalDialog, SettingsView
Models/        — BackendConfig, GlazingJob, Settings
```

## Configuration

Settings stored in `~/Library/Application Support/GlazeGuard/config.json`. Backend resolved from `backends.json` in project root. Glazing strength: Low (100 iters), Medium (200), High (500), Maximum (1000).
