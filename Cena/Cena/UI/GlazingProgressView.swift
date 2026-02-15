//
//  GlazingProgressView.swift
//  Cena
//
//  Animated progress view for the overlay window
//

import SwiftUI

struct GlazingProgressView: View {
    @ObservedObject var job: GlazingJob
    let onCancel: () -> Void

    @State private var animationAmount: CGFloat = 1.0

    var body: some View {
        ZStack {
            // Translucent background with vibrancy
            VisualEffectViewRepresentable(material: .hudWindow, blendingMode: .behindWindow)
                .cornerRadius(24)
                .shadow(color: .black.opacity(0.3), radius: 20, x: 0, y: 10)

            VStack(spacing: 28) {
                // Animated shield icon
                VStack(spacing: 16) {
                    ZStack {
                        // Pulsing circle background
                        Circle()
                            .fill(
                                LinearGradient(
                                    colors: [Color.blue.opacity(0.3), Color.purple.opacity(0.3)],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                            .frame(width: 120, height: 120)
                            .scaleEffect(animationAmount)
                            .opacity(2 - Double(animationAmount))
                            .animation(
                                Animation.easeOut(duration: 1.5)
                                    .repeatForever(autoreverses: false),
                                value: animationAmount
                            )

                        // Shield icon
                        Image(systemName: "shield.checkered")
                            .font(.system(size: 64))
                            .foregroundStyle(
                                LinearGradient(
                                    colors: [.blue, .purple],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                )
                            )
                            .symbolEffect(.pulse)
                    }

                    Text(statusText)
                        .font(.title2.weight(.semibold))
                        .foregroundColor(.white)
                }

                // Progress information
                VStack(spacing: 16) {
                    // Progress bar
                    VStack(spacing: 10) {
                        HStack {
                            Text("\(Int(job.progress * 100))%")
                                .font(.headline)
                                .foregroundColor(.white)

                            Spacer()

                            Text("\(job.currentIteration) / \(job.totalIterations)")
                                .font(.subheadline)
                                .foregroundColor(.white.opacity(0.8))
                        }

                        // Custom progress bar with gradient
                        GeometryReader { geometry in
                            ZStack(alignment: .leading) {
                                // Background
                                RoundedRectangle(cornerRadius: 8)
                                    .fill(Color.white.opacity(0.2))
                                    .frame(height: 12)

                                // Progress fill
                                RoundedRectangle(cornerRadius: 8)
                                    .fill(
                                        LinearGradient(
                                            colors: [.blue, .purple],
                                            startPoint: .leading,
                                            endPoint: .trailing
                                        )
                                    )
                                    .frame(width: geometry.size.width * CGFloat(job.progress), height: 12)
                                    .animation(.easeInOut(duration: 0.3), value: job.progress)
                            }
                        }
                        .frame(height: 12)

                        // Time remaining
                        HStack {
                            Image(systemName: "clock")
                                .font(.caption)

                            Text(job.estimatedTimeRemaining())
                                .font(.caption)
                        }
                        .foregroundColor(.white.opacity(0.7))
                    }
                    .frame(maxWidth: 360)

                    // Status message
                    if let statusMessage = getStatusMessage() {
                        Text(statusMessage)
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.6))
                            .multilineTextAlignment(.center)
                    }
                }

                // Cancel button
                Button(action: onCancel) {
                    Text("Cancel")
                        .font(.body.weight(.medium))
                        .foregroundColor(.white)
                        .frame(width: 140)
                        .padding(.vertical, 10)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color.red.opacity(0.7))
                        )
                }
                .buttonStyle(.plain)
                .onHover { hovering in
                    // TODO: Add hover effect
                }
            }
            .padding(44)
        }
        .frame(width: 450, height: 350)
        .onAppear {
            animationAmount = 1.5
        }
    }

    // MARK: - Helpers

    private var statusText: String {
        switch job.status {
        case .pending:
            return "Preparing..."
        case .generatingMask:
            return "Detecting Faces"
        case .protecting:
            return "Encrypting Your Likeness"
        case .completed:
            return "Encryption Complete!"
        case .failed:
            return "Encryption Failed"
        case .cancelled:
            return "Cancelled"
        }
    }

    private func getStatusMessage() -> String? {
        switch job.status {
        case .generatingMask:
            return "Using Vision framework to detect faces"
        case .protecting where job.currentIteration > 0:
            return "Applying adversarial perturbations"
        case .completed:
            return "Likeness encrypted and ready to use"
        case .failed(let error):
            return error.localizedDescription
        default:
            return nil
        }
    }
}

// MARK: - Preview

#if DEBUG
struct GlazingProgressView_Previews: PreviewProvider {
    static var previews: some View {
        let job = GlazingJob(
            image: NSImage(systemSymbolName: "photo", accessibilityDescription: nil) ?? NSImage(),
            iterations: 200
        )
        job.status = .protecting
        job.updateProgress(iteration: 75)
        job.startTime = Date().addingTimeInterval(-30)

        return GlazingProgressView(job: job, onCancel: {})
            .frame(width: 450, height: 350)
    }
}
#endif
