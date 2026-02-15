//
// EncryptionQueue.swift
// Cena
//
// queue of encryption jobs and coordinates processing
//

import Foundation
import AppKit

@MainActor
class EncryptionQueue: ObservableObject {
    @Published var jobs: [EncryptionJob] = []
    @Published var currentJob: EncryptionJob?
    @Published var isProcessing = false

    private let backendService: BackendService
    private let settings: Settings

    init(backendService: BackendService, settings: Settings) {
        self.backendService = backendService
        self.settings = settings
    }

    //MARK: - Queue Management

    /// add a new job to the queue and start processing
    func enqueue(_ job: EncryptionJob) {
        jobs.append(job)
        print("📥 Enqueued job \(job.id) - Queue size: \(jobs.count)")

        if !isProcessing {
            Task {
                await processQueue()
            }
        }
    }

    /// remove a completed or failed job from the queue
    func remove(_ job: EncryptionJob) {
        jobs.removeAll { $0.id == job.id }
        print("🗑️  Removed job \(job.id) - Queue size: \(jobs.count)")
    }

    /// clear all completed jobs
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

    //MARK: - Processing

    /// process jobs in the queue sequentially
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

    /// process a single encryption job
    private func processJob(_ job: EncryptionJob) async {
        print("⚙️  Processing job \(job.id)")

        job.startTime = Date()

        do {
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

            guard MaskGenerator.validateMask(mask) else {
                throw EncryptionError.maskGenerationFailed
            }

            job.status = .protecting

            let progressTask = Task {
                var iteration = 0
                let totalIterations = job.totalIterations
                let estimatedTimePerIter = 0.4

                while iteration < totalIterations && !Task.isCancelled {
                    try? await Task.sleep(nanoseconds: UInt64(estimatedTimePerIter * 1_000_000_000))
                    iteration += 1

                    await MainActor.run {
                        job.updateProgress(iteration: iteration)
                    }
                }
            }

            let rawProtected = try await backendService.protectImage(
                image: job.originalImage,
                mask: mask,
                iterations: job.iterations
            )

            let blended = job.intensity < 0.99
                ? rawProtected.blendWith(original: job.originalImage, alpha: job.intensity)
                : rawProtected

            let finalImage = blended.applyFinalization()

            progressTask.cancel()

            await MainActor.run {
                job.complete(with: finalImage)
            }

            print("✅ Job \(job.id) completed successfully")

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

    /// cancel a specific job
    func cancel(_ job: EncryptionJob) {
        job.cancel()
        print("🚫 Cancelled job \(job.id)")

        if currentJob?.id == job.id {
            currentJob = nil
        }
    }

    /// cancel all pending jobs
    func cancelAll() {
        for job in jobs where job.isPending {
            job.cancel()
        }
        currentJob = nil
        isProcessing = false
        print("🚫 Cancelled all jobs")
    }

    //MARK: - Notifications

    /// show system notification for completed job
    private func showNotification(for job: EncryptionJob) {
        let notification = NSUserNotification()
        notification.title = "Likeness Encrypted"
        notification.informativeText = "Your likeness has been encrypted and is ready to use"
        notification.soundName = NSUserNotificationDefaultSoundName

        NSUserNotificationCenter.default.deliver(notification)
    }

    //MARK: - Statistics

    /// queue stats
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
