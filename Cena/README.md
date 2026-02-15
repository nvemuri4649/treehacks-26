# Cena — Automatic Likeness Encryption

Native macOS menu bar app that automatically encrypts your likeness in images copied to clipboard, preventing deepfake generation. Shows a translucent overlay with progress while the encryption runs on the GPU backend.

## How It Works

1. Copy an image (Cmd+C) — Cena detects it via clipboard monitoring
2. Approval dialog appears (or auto-approve if configured)
3. Translucent overlay shows encryption progress
4. Clipboard is replaced with the encrypted image — paste as normal

Also supports drag-and-drop onto the menu bar icon, or "Protect Image..." from the menu.

## Build & Run

```bash
cd Cena
swift build -c release          # or open Package.swift in Xcode, Cmd+R
```

Requires macOS 13+ and a running encryption backend (see root README).

## Structure

```
App/           — Entry point, AppState, MenuBarController
Services/      — BackendService (HTTP), PasteboardMonitor, MaskGenerator (Vision), EncryptionQueue
UI/            — OverlayWindow (translucent HUD), EncryptionProgressView, ApprovalDialog, SettingsView
Models/        — BackendConfig, EncryptionJob, Settings
```

## Configuration

Settings stored in `~/Library/Application Support/Cena/config.json`. Backend resolved from `backends.json` in project root. Encryption strength: Low (100 iters), Medium (200), High (500), Maximum (1000).
