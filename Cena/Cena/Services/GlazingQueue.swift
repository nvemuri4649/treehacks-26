//
//  GlazingQueue.swift
//  Cena
//
//  Manages a queue of glazing jobs and coordinates processing
//

import Foundation
import AppKit

@MainActor
class GlazingQueue: ObservableObject {
    @Published var jobs: [GlazingJob] = []
    @Published var currentJob: GlazingJob?
    @Published var isProcessing = false

    private let backendService: BackendService
    private let settings: Settings

    init(backendService: BackendService, settings: Settings) {
        self.backendService = backendService
        self.settings = settings
    }

    // MARK: - Queue Management

    /// Add a new job to the queue and start processing
    func enqueue(_ job: GlazingJob) {
        jobs.append(job)
        print("📥 Enqueued job \(job.id) - Queue size: \(jobs.count)")

        // Start processing if not already running
        if !isProcessing {
            Task {
                await processQueue()
            }
        }
    }

    /// Remove a completed or failed job from the queue
    func remove(_ job: GlazingJob) {
        jobs.removeAll { $0.id == job.id }
        print("🗑️  Removed job \(job.id) - Queue size: \(jobs.count)")
    }

    /// Clear all completed jobs
    func clearCompleted() {
        let beforeCount = jobs.count
        jobs.removeAll { job in
            switch job.status {
            case .completed, .failed, .cancelled:
                return true
            default:
                return false
            }
        }
        let removedCount = beforeCount - jobs.count
        if removedCount > 0 {
            print("🧹 Cleared \(removedCount) completed job(s)")
        }
    }

    // MARK: - Processing

    /// Process jobs in the queue sequentially
    private func processQueue() async {
        guard !isProcessing else { return }
        isProcessing = true

        while let nextJob = jobs.first(where: { job in
            switch job.status {
            case .pending:
                return true
            default:
                return false
            }
        }) {
            currentJob = nextJob
            await processJob(nextJob)
            currentJob = nil
        }

        isProcessing = false
        print("✅ Queue processing complete")
    }

    /// Process a single glazing job
    private func processJob(_ job: GlazingJob) async {
        print("⚙️  Processing job \(job.id)")

        job.startTime = Date()

        do {
            // Step 1: Generate mask if needed
            let mask: NSImage
            switch job.maskMode {
            case .autoFace:
                job.status = .generatingMask
                mask = try await MaskGenerator.generateFaceMask(for: job.originalImage)
                job.maskImage = mask

            case .fullImage:
                mask = MaskGenerator.createFullMask(size: job.originalImage.size)
                job.maskImage = mask

            case .manual(let providedMask):
                mask = providedMask
                job.maskImage = mask
            }

            // Validate mask
            guard MaskGenerator.validateMask(mask) else {
                throw GlazingError.maskGenerationFailed
            }

            // Step 2: Protect image
            job.status = .protecting

            // Simulate progress updates (since we don't have real-time feedback yet)
            // TODO: Add SSE endpoint for real-time progress from server
            let progressTask = Task {
                var iteration = 0
                let totalIterations = job.totalIterations
                let estimatedTimePerIter = 0.4  // seconds

                while iteration < totalIterations && !Task.isCancelled {
                    try? await Task.sleep(nanoseconds: UInt64(estimatedTimePerIter * 1_000_000_000))
                    iteration += 1

                    await MainActor.run {
                        job.updateProgress(iteration: iteration)
                    }
                }
            }

            // Call backend service
            let protectedImage = try await backendService.protectImage(
                image: job.originalImage,
                mask: mask,
                iterations: job.iterations
            )

            // Cancel progress simulation
            progressTask.cancel()

            // Mark as complete
            await MainActor.run {
                job.complete(with: protectedImage)
            }

            print("✅ Job \(job.id) completed successfully")

            // Show notification if enabled
            if settings.showNotifications {
                showNotification(for: job)
            }

        } catch {
            print("❌ Job \(job.id) failed: \(error)")
            await MainActor.run {
                job.fail(with: error)
            }
        }
    }

    /// Cancel a specific job
    func cancel(_ job: GlazingJob) {
        job.cancel()
        print("🚫 Cancelled job \(job.id)")

        // If this was the current job, move to next
        if currentJob?.id == job.id {
            currentJob = nil
        }
    }

    /// Cancel all pending jobs
    func cancelAll() {
        for job in jobs where job.isPending {
            job.cancel()
        }
        currentJob = nil
        isProcessing = false
        print("🚫 Cancelled all jobs")
    }

    // MARK: - Notifications

    /// Show system notification for completed job
    private func showNotification(for job: GlazingJob) {
        let notification = NSUserNotification()
        notification.title = "Image Protected"
        notification.informativeText = "Your image has been glazed and is ready to use"
        notification.soundName = NSUserNotificationDefaultSoundName

        NSUserNotificationCenter.default.deliver(notification)
    }

    // MARK: - Statistics

    /// Get queue statistics
    func getStats() -> (pending: Int, processing: Int, completed: Int, failed: Int) {
        var pending = 0
        var processing = 0
        var completed = 0
        var failed = 0

        for job in jobs {
            switch job.status {
            case .pending:
                pending += 1
            case .generatingMask, .protecting:
                processing += 1
            case .completed:
                completed += 1
            case .failed:
                failed += 1
            case .cancelled:
                break
            }
        }

        return (pending, processing, completed, failed)
    }
}
