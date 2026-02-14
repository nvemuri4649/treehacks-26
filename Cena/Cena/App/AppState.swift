//
//  AppState.swift
//  Cena
//
//  Shared application state management
//

import Foundation
import SwiftUI
import AppKit

@MainActor
class AppState: ObservableObject {
    // Configuration
    @Published var settings: Settings
    @Published var backendConfig: BackendConfig?

    // Services
    let pasteboardMonitor: PasteboardMonitor
    var backendService: BackendService?
    var glazingQueue: GlazingQueue?

    // UI State
    @Published var showSettingsWindow = false
    @Published var showApprovalDialog = false
    @Published var overlayWindow: OverlayWindow?
    @Published var currentPendingImage: NSImage?

    // Status
    @Published var backendStatus: BackendStatus = .unknown
    @Published var lastError: String?

    enum BackendStatus {
        case unknown
        case checking
        case available
        case unavailable
    }

    init() {
        self.settings = Settings()
        self.pasteboardMonitor = PasteboardMonitor()

        // Load backend configuration
        if let config = BackendConfig.load() {
            self.backendConfig = config
            setupBackendService(config: config)
        } else {
            print("⚠️  Failed to load backend config")
        }
    }

    // MARK: - Setup

    func setupBackendService(config: BackendConfig) {
        guard let backend = config.getBackend(named: settings.selectedBackend) else {
            print("❌ Backend '\(settings.selectedBackend)' not found")
            return
        }

        print("🔧 Setting up backend service: \(settings.selectedBackend)")
        self.backendService = BackendService(baseURL: backend.url)
        self.glazingQueue = GlazingQueue(
            backendService: backendService!,
            settings: settings
        )

        // Check backend health
        Task {
            await checkBackendHealth()
        }
    }

    // MARK: - Backend Management

    func checkBackendHealth() async {
        guard let backendService = backendService else {
            backendStatus = .unavailable
            return
        }

        backendStatus = .checking

        do {
            let health = try await backendService.checkHealth()
            backendStatus = .available
            print("✅ Backend is healthy: \(health.status)")
        } catch {
            backendStatus = .unavailable
            lastError = "Backend unavailable: \(error.localizedDescription)"
            print("❌ Backend health check failed: \(error)")
        }
    }

    func changeBackend(to name: String) {
        guard let config = backendConfig,
              let backend = config.getBackend(named: name) else {
            print("❌ Backend '\(name)' not found")
            return
        }

        settings.selectedBackend = name
        settings.save()

        print("🔄 Switching to backend: \(name)")
        self.backendService = BackendService(baseURL: backend.url)
        self.glazingQueue = GlazingQueue(
            backendService: backendService!,
            settings: settings
        )

        Task {
            await checkBackendHealth()
        }
    }

    // MARK: - Pasteboard Monitoring

    func startPasteboardMonitoring() {
        guard settings.monitorPasteboard else {
            print("⚠️  Pasteboard monitoring is disabled in settings")
            return
        }

        pasteboardMonitor.startMonitoring { [weak self] image in
            Task { @MainActor in
                self?.handleImageCopied(image)
            }
        }
    }

    func stopPasteboardMonitoring() {
        pasteboardMonitor.stopMonitoring()
    }

    private func handleImageCopied(_ image: NSImage) {
        guard settings.enabled else { return }

        print("📋 Image copied to pasteboard")

        if settings.autoApprove {
            // Auto-approve: glaze immediately
            startGlazingJob(
                image: image,
                iterations: settings.defaultIterations,
                maskMode: settings.getMaskMode(),
                source: .pasteboard
            )
        } else {
            // Show approval dialog
            currentPendingImage = image
            showApprovalDialog = true
        }
    }

    // MARK: - Glazing Operations

    func startGlazingJob(
        image: NSImage,
        iterations: Int,
        maskMode: GlazingJob.MaskMode,
        source: GlazingJob.JobSource
    ) {
        guard let glazingQueue = glazingQueue else {
            print("❌ Glazing queue not initialized")
            return
        }

        let job = GlazingJob(
            image: image,
            iterations: iterations,
            maskMode: maskMode,
            source: source
        )

        // Show overlay
        showOverlay(for: job)

        // Enqueue job
        glazingQueue.enqueue(job)

        // Monitor job completion
        monitorJob(job)
    }

    private func monitorJob(_ job: GlazingJob) {
        // Poll job status
        Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self, weak job] timer in
            guard let self = self, let job = job else {
                timer.invalidate()
                return
            }

            Task { @MainActor in
                switch job.status {
                case .completed:
                    timer.invalidate()
                    self.handleJobCompleted(job)
                case .failed(let error):
                    timer.invalidate()
                    self.handleJobFailed(job, error: error)
                case .cancelled:
                    timer.invalidate()
                    self.handleJobCancelled(job)
                default:
                    break
                }
            }
        }
    }

    private func handleJobCompleted(_ job: GlazingJob) {
        print("✅ Job completed: \(job.id)")

        // Replace pasteboard if job was from clipboard
        if job.source == .pasteboard, let protectedImage = job.protectedImage {
            pasteboardMonitor.replacePasteboardImage(with: protectedImage)
        }

        // Hide overlay
        overlayWindow?.hide {
            self.overlayWindow = nil
        }

        // Show notification
        if settings.showNotifications {
            showCompletionNotification(success: true)
        }
    }

    private func handleJobFailed(_ job: GlazingJob, error: Error) {
        print("❌ Job failed: \(job.id) - \(error)")
        lastError = error.localizedDescription

        overlayWindow?.hide {
            self.overlayWindow = nil
        }

        if settings.showNotifications {
            showCompletionNotification(success: false, error: error.localizedDescription)
        }
    }

    private func handleJobCancelled(_ job: GlazingJob) {
        print("🚫 Job cancelled: \(job.id)")

        overlayWindow?.hide {
            self.overlayWindow = nil
        }
    }

    // MARK: - UI Management

    private func showOverlay(for job: GlazingJob) {
        let progressView = GlazingProgressView(job: job) { [weak self] in
            self?.glazingQueue?.cancel(job)
        }

        let hostingView = NSHostingView(rootView: progressView)
        let window = OverlayWindow(contentView: hostingView)

        overlayWindow = window
        window.show()
    }

    private func showCompletionNotification(success: Bool, error: String? = nil) {
        let notification = NSUserNotification()

        if success {
            notification.title = "✅ Image Protected"
            notification.informativeText = "Your image has been glazed and is ready to use"
            notification.soundName = NSUserNotificationDefaultSoundName
        } else {
            notification.title = "❌ Protection Failed"
            notification.informativeText = error ?? "An unknown error occurred"
        }

        NSUserNotificationCenter.default.deliver(notification)
    }

    // MARK: - File Operations

    func protectImageFromFile(url: URL) {
        guard let image = NSImage(contentsOf: url) else {
            print("❌ Failed to load image from: \(url.path)")
            return
        }

        startGlazingJob(
            image: image,
            iterations: settings.defaultIterations,
            maskMode: settings.getMaskMode(),
            source: .filePicker
        )
    }

    func saveProtectedImage(_ image: NSImage, originalURL: URL? = nil) {
        let savePanel = NSSavePanel()
        savePanel.allowedContentTypes = [.png]
        savePanel.canCreateDirectories = true
        savePanel.isExtensionHidden = false

        if let originalURL = originalURL {
            let filename = originalURL.deletingPathExtension().lastPathComponent
            savePanel.nameFieldStringValue = "\(filename)_protected.png"
            savePanel.directoryURL = originalURL.deletingLastPathComponent()
        } else {
            savePanel.nameFieldStringValue = "protected_image.png"
        }

        savePanel.begin { [weak self] response in
            guard response == .OK, let url = savePanel.url else { return }

            if let pngData = image.pngData() {
                do {
                    try pngData.write(to: url)
                    print("💾 Saved protected image to: \(url.path)")

                    if self?.settings.showNotifications == true {
                        let notification = NSUserNotification()
                        notification.title = "Image Saved"
                        notification.informativeText = "Protected image saved to \(url.lastPathComponent)"
                        NSUserNotificationCenter.default.deliver(notification)
                    }
                } catch {
                    print("❌ Failed to save image: \(error)")
                    self?.lastError = "Failed to save image: \(error.localizedDescription)"
                }
            }
        }
    }
}
