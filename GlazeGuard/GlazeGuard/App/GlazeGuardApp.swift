//
//  GlazeGuardApp.swift
//  GlazeGuard
//
//  Main application entry point
//

import SwiftUI
import AppKit

@main
struct GlazeGuardApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        // Empty scene - we only use menu bar
        Settings {
            EmptyView()
        }
    }
}

class AppDelegate: NSObject, NSApplicationDelegate, NSUserNotificationCenterDelegate {
    private var menuBarController: MenuBarController!
    private var appState: AppState!
    private var approvalWindowController: NSWindowController?

    func applicationDidFinishLaunching(_ notification: Notification) {
        print("🚀 GlazeGuard starting...")

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

        print("✅ GlazeGuard ready")
        print("   Backend: \(appState.settings.selectedBackend)")
        print("   Monitoring: \(appState.settings.monitorPasteboard)")
    }

    func applicationWillTerminate(_ notification: Notification) {
        print("👋 GlazeGuard shutting down...")
        appState.stopPasteboardMonitoring()
    }

    // MARK: - Approval Dialog

    private func setupApprovalDialogObserver() {
        // Observe changes to showApprovalDialog
        Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self = self else { return }

            if self.appState.showApprovalDialog,
               let image = self.appState.currentPendingImage,
               self.approvalWindowController == nil {

                self.showApprovalDialog(for: image)
                self.appState.showApprovalDialog = false
            }
        }
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

        window.title = "GlazeGuard"
        window.styleMask = [.titled, .closable]
        window.level = .floating
        window.center()

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
        case .once(let iterations, let maskMode):
            let mode: GlazingJob.MaskMode = maskMode == "auto_face" ? .autoFace : .fullImage
            appState.startGlazingJob(
                image: image,
                iterations: iterations,
                maskMode: mode,
                source: .pasteboard
            )

        case .always(let iterations, let maskMode):
            // Save preference
            appState.settings.autoApprove = true
            appState.settings.defaultIterations = iterations
            appState.settings.defaultMaskMode = maskMode
            appState.settings.save()

            let mode: GlazingJob.MaskMode = maskMode == "auto_face" ? .autoFace : .fullImage
            appState.startGlazingJob(
                image: image,
                iterations: iterations,
                maskMode: mode,
                source: .pasteboard
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
