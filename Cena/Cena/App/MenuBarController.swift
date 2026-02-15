//
//  MenuBarController.swift
//  Cena
//
//  Menu bar app controller with status icon and menu
//

import AppKit
import SwiftUI

@MainActor
class MenuBarController: NSObject {
    private var statusItem: NSStatusItem!
    private var appState: AppState
    private var settingsWindow: NSWindow?
    private var approvalWindow: NSWindow?
    private var agentChatWindow: NSWindow?
    private var agentPipelineWindow: NSWindow?
    private var deepfakeScannerWindow: NSWindow?
    private var demoWindow: NSWindow?
    private var cinematicWindow: NSWindow?
    private let pipelineModel = PipelineStageModel()

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
            button.image = NSImage(systemSymbolName: "shield.checkered", accessibilityDescription: "Cena")
            button.image?.isTemplate = true

            // Enable drag and drop
            button.window?.registerForDraggedTypes([.fileURL, .png, .tiff])
            button.window?.delegate = self
        }
    }

    private func setupMenu() {
        let menu = NSMenu()

        // Status info
        let titleItem = NSMenuItem(
            title: "Cena",
            action: nil,
            keyEquivalent: ""
        )
        titleItem.isEnabled = false
        menu.addItem(titleItem)

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
            title: appState.settings.enabled ? "Disable Cena" : "Enable Cena",
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

        // Agent Chat
        let agentItem = NSMenuItem(
            title: "Open Agent Chat...",
            action: #selector(openAgentChat),
            keyEquivalent: "a"
        )
        menu.addItem(agentItem)

        // Protect image from file
        let protectItem = NSMenuItem(
            title: "Encrypt Likeness...",
            action: #selector(protectFromFile),
            keyEquivalent: "o"
        )
        menu.addItem(protectItem)

        // Deepfake detection scanner
        let deepfakeItem = NSMenuItem(
            title: "Scan for Deepfakes...",
            action: #selector(openDeepfakeScanner),
            keyEquivalent: "d"
        )
        menu.addItem(deepfakeItem)

        // Demo
        let demoItem = NSMenuItem(
            title: "Encryption Demo...",
            action: #selector(openDemo),
            keyEquivalent: "t"
        )
        menu.addItem(demoItem)

        // Watermark robustness demo
        let cinematicItem = NSMenuItem(
            title: "Watermark Robustness...",
            action: #selector(openCinematic),
            keyEquivalent: "i"
        )
        menu.addItem(cinematicItem)

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
            title: "Quit Cena",
            action: #selector(quit),
            keyEquivalent: "q"
        )
        menu.addItem(quitItem)

        // Set target on all actionable items so selectors resolve
        for item in menu.items where item.action != nil {
            item.target = self
        }

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
            menu.items[4].title = appState.settings.enabled ? "Disable Cena" : "Enable Cena"
        }

        // Update monitoring toggle (item at index 5)
        if menu.items.count > 5 {
            menu.items[5].title = appState.pasteboardMonitor.isMonitoring ? "Stop Monitoring Clipboard" : "Start Monitoring Clipboard"
        }
    }

    private func backendStatusText() -> String {
        let backendName = appState.settings.selectedBackend.capitalized
        return "Backend: \(backendName)"
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

        print(appState.settings.enabled ? "✅ Cena enabled" : "⏸️  Cena disabled")
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

    @objc private func openAgentChat() {
        if let window = agentChatWindow, window.isVisible {
            window.makeKeyAndOrderFront(nil)
            agentPipelineWindow?.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        // ── Chat window (right) ───────────────────────────────────
        let chatView = AgentChatView(pipelineModel: pipelineModel)
        let hostingController = NSHostingController(rootView: chatView)

        let chatWindow = NSWindow(contentViewController: hostingController)
        chatWindow.title = "Cena"
        chatWindow.titlebarAppearsTransparent = true
        chatWindow.titleVisibility = .hidden
        chatWindow.styleMask = [.titled, .closable, .resizable, .miniaturizable, .fullSizeContentView]
        chatWindow.setContentSize(NSSize(width: 720, height: 640))
        chatWindow.minSize = NSSize(width: 520, height: 360)
        chatWindow.isReleasedWhenClosed = false
        chatWindow.isOpaque = false
        chatWindow.backgroundColor = .clear
        chatWindow.hasShadow = true

        // ── Pipeline window (left companion) ──────────────────────
        let pipelineView = AgentPipelineWindowView(model: pipelineModel)
        let pipelineHosting = NSHostingController(rootView: pipelineView)

        let pipeWindow = NSWindow(contentViewController: pipelineHosting)
        pipeWindow.title = "Pipeline"
        pipeWindow.titlebarAppearsTransparent = true
        pipeWindow.titleVisibility = .hidden
        pipeWindow.styleMask = [.titled, .closable, .fullSizeContentView]
        pipeWindow.setContentSize(NSSize(width: 260, height: 640))
        pipeWindow.isReleasedWhenClosed = false
        pipeWindow.isOpaque = false
        pipeWindow.backgroundColor = .clear
        pipeWindow.hasShadow = true

        // Position: pipeline window on left, chat window adjacent to its right
        if let screen = NSScreen.main {
            let vis = screen.visibleFrame
            let gap: CGFloat = 8
            let totalWidth = 260 + gap + 720
            let startX = vis.origin.x + (vis.width - totalWidth) / 2
            let centerY = vis.origin.y + (vis.height - 640) / 2

            pipeWindow.setFrameOrigin(NSPoint(x: startX, y: centerY))
            chatWindow.setFrameOrigin(NSPoint(x: startX + 260 + gap, y: centerY))
        } else {
            chatWindow.center()
            pipeWindow.setFrameOrigin(NSPoint(
                x: chatWindow.frame.origin.x - 260 - 8,
                y: chatWindow.frame.origin.y
            ))
        }

        pipeWindow.makeKeyAndOrderFront(nil)
        chatWindow.makeKeyAndOrderFront(nil)

        agentChatWindow = chatWindow
        agentPipelineWindow = pipeWindow
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func openDeepfakeScanner() {
        if let window = deepfakeScannerWindow, window.isVisible {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let scannerView = DeepfakeScannerView()
        let hostingController = NSHostingController(rootView: scannerView)

        let window = NSWindow(contentViewController: hostingController)
        window.title = "Deepfake Scanner"
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.styleMask = [.titled, .closable, .resizable, .miniaturizable, .fullSizeContentView]
        window.setContentSize(NSSize(width: 820, height: 720))
        window.minSize = NSSize(width: 700, height: 600)
        window.isReleasedWhenClosed = false
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = true

        Self.positionWindowLeftMiddle(window)
        window.makeKeyAndOrderFront(nil)

        deepfakeScannerWindow = window
        NSApp.activate(ignoringOtherApps: true)
    }

    // MARK: - Window positioning helpers

    /// Position a window at the left-middle of the main screen with comfortable margin.
    static func positionWindowLeftMiddle(_ window: NSWindow, marginX: CGFloat = 40) {
        guard let screen = NSScreen.main else { return }
        let screenFrame = screen.visibleFrame
        let windowSize = window.frame.size
        let x = screenFrame.origin.x + marginX
        let y = screenFrame.origin.y + (screenFrame.height - windowSize.height) / 2
        window.setFrameOrigin(NSPoint(x: x, y: y))
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

    @objc private func openCinematic() {
        if let window = cinematicWindow, window.isVisible {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let view = CinematicDisplayView()
        let hc = NSHostingController(rootView: view)

        let window = NSWindow(contentViewController: hc)
        window.title = "Watermark Robustness"
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.styleMask = [.titled, .closable, .fullSizeContentView]
        window.setContentSize(NSSize(width: 580, height: 850))
        window.isReleasedWhenClosed = false
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = true

        Self.positionWindowLeftMiddle(window)
        window.makeKeyAndOrderFront(nil)

        cinematicWindow = window
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func openDemo() {
        if let window = demoWindow, window.isVisible {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let demoView = DemoView()
        let hostingController = NSHostingController(rootView: demoView)

        let window = NSWindow(contentViewController: hostingController)
        window.title = "Demo"
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.styleMask = [.titled, .closable, .resizable, .miniaturizable, .fullSizeContentView]
        window.setContentSize(NSSize(width: 860, height: 700))
        window.minSize = NSSize(width: 700, height: 550)
        window.isReleasedWhenClosed = false
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = true
        window.center()
        window.makeKeyAndOrderFront(nil)

        demoWindow = window
        NSApp.activate(ignoringOtherApps: true)
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
        window.title = "Settings"
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.styleMask = [.titled, .closable, .fullSizeContentView]
        window.setContentSize(NSSize(width: 480, height: 520))
        window.center()
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = true
        window.makeKeyAndOrderFront(nil)
        window.isReleasedWhenClosed = false

        // Apply backend changes when window closes
        NotificationCenter.default.addObserver(
            forName: NSWindow.willCloseNotification,
            object: window,
            queue: .main
        ) { [weak self] _ in
            if let currentBackend = self?.appState.settings.selectedBackend {
                self?.appState.changeBackend(to: currentBackend)
            }
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

@MainActor
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
