//
//  MenuBarController.swift
//  GlazeGuard
//
//  Menu bar app controller with status icon and menu
//

import AppKit
import SwiftUI

class MenuBarController: NSObject {
    private var statusItem: NSStatusItem!
    private var appState: AppState
    private var settingsWindow: NSWindow?
    private var approvalWindow: NSWindow?

    init(appState: AppState) {
        self.appState = appState
        super.init()

        setupStatusItem()
        setupMenu()
    }

    // MARK: - Status Item Setup

    private func setupStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)

        if let button = statusItem.button {
            // Use shield icon
            button.image = NSImage(systemSymbolName: "shield.checkered", accessibilityDescription: "GlazeGuard")
            button.image?.isTemplate = true

            // Enable drag and drop
            button.window?.registerForDraggedTypes([.fileURL, .png, .tiff])
            button.window?.delegate = self
        }
    }

    private func setupMenu() {
        let menu = NSMenu()

        // Status info
        let statusItem = NSMenuItem(
            title: "GlazeGuard",
            action: nil,
            keyEquivalent: ""
        )
        statusItem.isEnabled = false
        menu.addItem(statusItem)

        menu.addItem(NSMenuItem.separator())

        // Backend status
        let backendMenuItem = NSMenuItem(
            title: backendStatusText(),
            action: #selector(checkBackend),
            keyEquivalent: ""
        )
        menu.addItem(backendMenuItem)

        menu.addItem(NSMenuItem.separator())

        // Enable/Disable
        let enableItem = NSMenuItem(
            title: appState.settings.enabled ? "Disable GlazeGuard" : "Enable GlazeGuard",
            action: #selector(toggleEnabled),
            keyEquivalent: "e"
        )
        menu.addItem(enableItem)

        // Pasteboard monitoring toggle
        let monitorItem = NSMenuItem(
            title: appState.settings.monitorPasteboard ? "Stop Monitoring Clipboard" : "Start Monitoring Clipboard",
            action: #selector(toggleMonitoring),
            keyEquivalent: "m"
        )
        menu.addItem(monitorItem)

        menu.addItem(NSMenuItem.separator())

        // Protect image from file
        let protectItem = NSMenuItem(
            title: "Protect Image...",
            action: #selector(protectFromFile),
            keyEquivalent: "o"
        )
        menu.addItem(protectItem)

        menu.addItem(NSMenuItem.separator())

        // Settings
        let settingsItem = NSMenuItem(
            title: "Settings...",
            action: #selector(showSettings),
            keyEquivalent: ","
        )
        menu.addItem(settingsItem)

        menu.addItem(NSMenuItem.separator())

        // Quit
        let quitItem = NSMenuItem(
            title: "Quit GlazeGuard",
            action: #selector(quit),
            keyEquivalent: "q"
        )
        menu.addItem(quitItem)

        statusItem.menu = menu

        // Update menu periodically to reflect state changes
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.updateMenu()
        }
    }

    private func updateMenu() {
        guard let menu = statusItem.menu else { return }

        // Update backend status (item at index 2)
        if menu.items.count > 2 {
            menu.items[2].title = backendStatusText()
        }

        // Update enable/disable (item at index 4)
        if menu.items.count > 4 {
            menu.items[4].title = appState.settings.enabled ? "Disable GlazeGuard" : "Enable GlazeGuard"
        }

        // Update monitoring toggle (item at index 5)
        if menu.items.count > 5 {
            menu.items[5].title = appState.pasteboardMonitor.isMonitoring ? "Stop Monitoring Clipboard" : "Start Monitoring Clipboard"
        }
    }

    private func backendStatusText() -> String {
        let backendName = appState.settings.selectedBackend.capitalized

        switch appState.backendStatus {
        case .available:
            return "✅ Backend: \(backendName)"
        case .unavailable:
            return "❌ Backend: \(backendName) (unavailable)"
        case .checking:
            return "⏳ Checking backend..."
        case .unknown:
            return "❓ Backend: \(backendName)"
        }
    }

    // MARK: - Menu Actions

    @objc private func toggleEnabled() {
        appState.settings.enabled.toggle()
        appState.settings.save()

        if !appState.settings.enabled {
            appState.stopPasteboardMonitoring()
        } else if appState.settings.monitorPasteboard {
            appState.startPasteboardMonitoring()
        }

        print(appState.settings.enabled ? "✅ GlazeGuard enabled" : "⏸️  GlazeGuard disabled")
    }

    @objc private func toggleMonitoring() {
        if appState.pasteboardMonitor.isMonitoring {
            appState.stopPasteboardMonitoring()
        } else {
            appState.startPasteboardMonitoring()
        }
    }

    @objc private func checkBackend() {
        Task { @MainActor in
            await appState.checkBackendHealth()
        }
    }

    @objc private func protectFromFile() {
        let openPanel = NSOpenPanel()
        openPanel.allowsMultipleSelection = false
        openPanel.canChooseDirectories = false
        openPanel.canChooseFiles = true
        openPanel.allowedContentTypes = [.png, .jpeg, .tiff, .bmp, .gif]
        openPanel.message = "Select an image to protect"

        openPanel.begin { [weak self] response in
            guard response == .OK, let url = openPanel.url else { return }

            Task { @MainActor in
                self?.appState.protectImageFromFile(url: url)
            }
        }
    }

    @objc private func showSettings() {
        if let window = settingsWindow, window.isVisible {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        guard let config = appState.backendConfig else {
            print("❌ Backend config not loaded")
            return
        }

        let settingsView = SettingsView(settings: appState.settings, backendConfig: config)
        let hostingController = NSHostingController(rootView: settingsView)

        let window = NSWindow(contentViewController: hostingController)
        window.title = "GlazeGuard Settings"
        window.styleMask = [.titled, .closable]
        window.center()
        window.makeKeyAndOrderFront(nil)
        window.isReleasedWhenClosed = false

        // Listen for backend changes
        window.contentViewController?.view.window?.windowController?.shouldCloseDocument = { [weak self] in
            // Apply backend changes
            if let currentBackend = self?.appState.settings.selectedBackend {
                self?.appState.changeBackend(to: currentBackend)
            }
            return true
        }

        settingsWindow = window
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func quit() {
        appState.stopPasteboardMonitoring()
        NSApplication.shared.terminate(nil)
    }
}

// MARK: - Drag and Drop Support

extension MenuBarController: NSWindowDelegate {
    func draggingEntered(_ sender: NSDraggingInfo) -> NSDragOperation {
        // Check if dragging contains image files
        if let _ = getImageURL(from: sender) {
            return .copy
        }
        return []
    }

    func performDragOperation(_ sender: NSDraggingInfo) -> Bool {
        guard let url = getImageURL(from: sender) else {
            return false
        }

        Task { @MainActor in
            appState.protectImageFromFile(url: url)
        }

        return true
    }

    private func getImageURL(from draggingInfo: NSDraggingInfo) -> URL? {
        let pasteboard = draggingInfo.draggingPasteboard

        guard let urls = pasteboard.readObjects(forClasses: [NSURL.self], options: nil) as? [URL],
              let url = urls.first else {
            return nil
        }

        // Check if it's an image file
        let imageExtensions = ["png", "jpg", "jpeg", "tiff", "tif", "gif", "bmp", "heic", "heif"]
        let ext = url.pathExtension.lowercased()

        return imageExtensions.contains(ext) ? url : nil
    }
}
