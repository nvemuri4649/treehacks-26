//
//  CenaApp.swift
//  Cena
//
//  Main application entry point
//

import SwiftUI
import AppKit

@main
struct CenaApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        // Menu bar only app — no main window
        Window("Cena", id: "main") {
            EmptyView()
                .frame(width: 0, height: 0)
        }
        .defaultSize(width: 0, height: 0)
    }
}

@MainActor
class AppDelegate: NSObject, NSApplicationDelegate, NSUserNotificationCenterDelegate {
    private var menuBarController: MenuBarController!
    private var appState: AppState!
    private var approvalWindowController: NSWindowController?

    func applicationDidFinishLaunching(_ notification: Notification) {
        print("🚀 Cena starting...")

        // Hide dock icon (menu bar app only)
        NSApp.setActivationPolicy(.accessory)

        // Initialize app state
        appState = AppState()

        // Setup menu bar controller
        menuBarController = MenuBarController(appState: appState)

        // Setup notification center
        NSUserNotificationCenter.default.delegate = self

        // Start pasteboard monitoring if enabled
        if appState.settings.enabled && appState.settings.monitorPasteboard {
            appState.startPasteboardMonitoring()
        }

        // Monitor for approval dialog
        setupApprovalDialogObserver()

        print("✅ Cena ready")
        print("   Backend: \(appState.settings.selectedBackend)")
        print("   Monitoring: \(appState.settings.monitorPasteboard)")
    }

    func applicationWillTerminate(_ notification: Notification) {
        print("👋 Cena shutting down...")
        appState.stopPasteboardMonitoring()
    }

    // MARK: - Approval Dialog

    private func setupApprovalDialogObserver() {
        // Poll on main RunLoop for approval dialog requests
        let t = Timer(timeInterval: 0.2, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self = self else { return }

                if self.appState.showApprovalDialog,
                   let image = self.appState.currentPendingImage,
                   self.approvalWindowController == nil {

                    print("🔐 Showing approval dialog")
                    self.showApprovalDialog(for: image)
                    self.appState.showApprovalDialog = false
                }
            }
        }
        RunLoop.main.add(t, forMode: .common)
    }

    private func showApprovalDialog(for image: NSImage) {
        let approvalView = ApprovalDialog(
            image: image,
            onApprove: { [weak self] option in
                self?.handleApproval(option, image: image)
            },
            onDeny: { [weak self] in
                self?.handleDenial()
            }
        )

        let hostingController = NSHostingController(rootView: approvalView)
        let window = NSWindow(contentViewController: hostingController)

        window.title = "Cena"
        window.titlebarAppearsTransparent = true
        window.titleVisibility = .hidden
        window.styleMask = [.titled, .closable, .fullSizeContentView]
        window.level = .floating
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = true

        // Position at left-middle of screen
        MenuBarController.positionWindowLeftMiddle(window)

        let windowController = NSWindowController(window: window)
        windowController.showWindow(nil)

        approvalWindowController = windowController

        NSApp.activate(ignoringOtherApps: true)

        // Auto-close when done
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            if let window = windowController.window, !window.isVisible {
                self.approvalWindowController = nil
            }
        }
    }

    private func handleApproval(_ option: ApprovalDialog.ApprovalOption, image: NSImage) {
        print("✅ User approved protection")

        approvalWindowController = nil

        switch option {
        case .once(let iterations, let maskMode, let intensity):
            let mode: GlazingJob.MaskMode = maskMode == "auto_face" ? .autoFace : .fullImage
            appState.startGlazingJob(
                image: image,
                iterations: iterations,
                maskMode: mode,
                source: .pasteboard,
                intensity: intensity
            )

        case .always(let iterations, let maskMode, let intensity):
            appState.settings.autoApprove = true
            appState.settings.defaultIterations = iterations
            appState.settings.defaultMaskMode = maskMode
            appState.settings.save()

            let mode: GlazingJob.MaskMode = maskMode == "auto_face" ? .autoFace : .fullImage
            appState.startGlazingJob(
                image: image,
                iterations: iterations,
                maskMode: mode,
                source: .pasteboard,
                intensity: intensity
            )
        }
    }

    private func handleDenial() {
        print("⛔ User denied protection")
        approvalWindowController = nil
        appState.currentPendingImage = nil
    }

    // MARK: - Notification Delegate

    func userNotificationCenter(_ center: NSUserNotificationCenter, shouldPresent notification: NSUserNotification) -> Bool {
        return true  // Show notification even if app is active
    }

    func userNotificationCenter(_ center: NSUserNotificationCenter, didActivate notification: NSUserNotification) {
        // Handle notification click if needed
    }
}
