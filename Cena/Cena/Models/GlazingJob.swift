//
//  GlazingJob.swift
//  Cena
//
//  Represents a single image glazing job and its state
//

import Foundation
import AppKit

enum GlazingJobStatus {
    case pending
    case generatingMask
    case protecting
    case completed
    case failed(Error)
    case cancelled
}

class GlazingJob: ObservableObject, Identifiable {
    let id = UUID()

    @Published var status: GlazingJobStatus = .pending
    @Published var progress: Double = 0.0  // 0.0 to 1.0
    @Published var currentIteration: Int = 0
    @Published var totalIterations: Int = 200
    @Published var startTime: Date?
    @Published var endTime: Date?

    let originalImage: NSImage
    let iterations: Int
    let maskMode: MaskMode
    let source: JobSource

    var protectedImage: NSImage?
    var maskImage: NSImage?

    enum MaskMode {
        case autoFace       // Automatically detect faces
        case fullImage      // Protect entire image
        case manual(NSImage) // Use provided mask
    }

    enum JobSource {
        case pasteboard     // From clipboard monitoring
        case dragDrop       // From drag-and-drop to menu bar
        case filePicker     // From manual file selection
    }

    init(image: NSImage, iterations: Int = 200, maskMode: MaskMode = .autoFace, source: JobSource = .pasteboard) {
        self.originalImage = image
        self.iterations = iterations
        self.totalIterations = iterations
        self.maskMode = maskMode
        self.source = source
    }

    /// Calculate estimated time remaining based on current progress
    func estimatedTimeRemaining() -> String {
        guard let startTime = startTime, progress > 0.01 else {
            return "Calculating..."
        }

        let elapsed = Date().timeIntervalSince(startTime)
        let totalEstimated = elapsed / progress
        let remaining = totalEstimated - elapsed

        if remaining < 60 {
            return "\(Int(remaining))s remaining"
        } else {
            let minutes = Int(remaining / 60)
            let seconds = Int(remaining.truncatingRemainder(dividingBy: 60))
            return "\(minutes)m \(seconds)s remaining"
        }
    }

    /// Update progress based on current iteration
    func updateProgress(iteration: Int) {
        self.currentIteration = iteration
        self.progress = Double(iteration) / Double(totalIterations)
    }

    /// Mark job as completed successfully
    func complete(with protectedImage: NSImage) {
        self.protectedImage = protectedImage
        self.status = .completed
        self.progress = 1.0
        self.endTime = Date()
    }

    /// Mark job as failed with error
    func fail(with error: Error) {
        self.status = .failed(error)
        self.endTime = Date()
    }

    /// Cancel the job
    func cancel() {
        self.status = .cancelled
        self.endTime = Date()
    }
}

enum GlazingError: LocalizedError {
    case invalidImage
    case invalidMask
    case invalidResponse
    case backendUnavailable
    case networkError(Error)
    case maskGenerationFailed
    case cancelled

    var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "Invalid or corrupted image"
        case .invalidMask:
            return "Invalid or corrupted mask"
        case .invalidResponse:
            return "Server returned invalid response"
        case .backendUnavailable:
            return "Backend server is unavailable"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .maskGenerationFailed:
            return "Failed to generate face detection mask"
        case .cancelled:
            return "Operation was cancelled"
        }
    }
}
