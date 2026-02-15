//
//  ApprovalDialog.swift
//  Cena
//
//  Dialog asking user for consent before encrypting a likeness
//

import SwiftUI

struct ApprovalDialog: View {
    let image: NSImage
    let onApprove: (ApprovalOption) -> Void
    let onDeny: () -> Void
    @Environment(\.dismiss) var dismiss

    @State private var selectedStrength: Int = 200
    @State private var selectedMaskMode: String = "auto_face"
    @State private var intensity: Double = 1.0
    @State private var rememberChoice = false

    enum ApprovalOption {
        case once(iterations: Int, maskMode: String, intensity: Double)
        case always(iterations: Int, maskMode: String, intensity: Double)
    }

    var body: some View {
        VStack(spacing: 20) {
            // Header
            HStack(spacing: 14) {
                CenaLogo(size: 48, isAnimating: false, color: .white.opacity(0.8))

                VStack(alignment: .leading, spacing: 4) {
                    Text("Encrypt This Likeness?")
                        .font(.title2.weight(.semibold))
                    Text("Automatic Likeness Encryption prevents deepfake generation")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                Spacer()
            }

            Divider()

            // Image preview + settings
            HStack(alignment: .top) {
                if let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) {
                    Image(decorative: cgImage, scale: 1.0)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 120, height: 120)
                        .cornerRadius(8)
                        .shadow(radius: 2)
                } else {
                    Image(systemName: "photo")
                        .font(.system(size: 60))
                        .foregroundColor(.secondary)
                        .frame(width: 120, height: 120)
                }

                VStack(alignment: .leading, spacing: 12) {
                    // Iterations
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Iterations:")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundColor(.secondary)

                        Picker("", selection: $selectedStrength) {
                            Text("50").tag(50)
                            Text("100").tag(100)
                            Text("200").tag(200)
                            Text("500").tag(500)
                            Text("1000").tag(1000)
                        }
                        .pickerStyle(.segmented)
                    }

                    // Region
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Region:")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundColor(.secondary)

                        Picker("", selection: $selectedMaskMode) {
                            Text("Faces").tag("auto_face")
                            Text("Full Image").tag("full_image")
                        }
                        .pickerStyle(.segmented)
                    }

                    // Intensity slider
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Intensity:")
                                .font(.system(size: 11, weight: .medium))
                                .foregroundColor(.secondary)
                            Spacer()
                            Text("\(Int(intensity * 100))%")
                                .font(.system(size: 11, weight: .bold, design: .rounded))
                                .foregroundColor(.white.opacity(0.7))
                        }

                        Slider(value: $intensity, in: 0.05...1.0, step: 0.05)
                            .tint(.purple)

                        HStack {
                            Text("Subtle")
                                .font(.system(size: 9))
                                .foregroundColor(.secondary.opacity(0.6))
                            Spacer()
                            Text("Full")
                                .font(.system(size: 9))
                                .foregroundColor(.secondary.opacity(0.6))
                        }
                    }

                    // Estimate
                    HStack {
                        Image(systemName: "clock")
                            .font(.caption)
                        Text("Est. time: \(estimatedTime)")
                            .font(.caption)
                    }
                    .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            Toggle("Always encrypt copied images", isOn: $rememberChoice)
                .toggleStyle(.checkbox)
                .font(.caption)

            Divider()

            // Buttons
            HStack {
                Button("Skip") {
                    onDeny()
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)

                Spacer()

                Button("Encrypt") {
                    let option: ApprovalOption
                    if rememberChoice {
                        option = .always(iterations: selectedStrength, maskMode: selectedMaskMode, intensity: intensity)
                    } else {
                        option = .once(iterations: selectedStrength, maskMode: selectedMaskMode, intensity: intensity)
                    }
                    onApprove(option)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(28)
        .frame(width: 580)
        .background(
            VisualEffectViewRepresentable(material: .hudWindow, blendingMode: .behindWindow)
                .ignoresSafeArea()
        )
    }

    private var estimatedTime: String {
        let seconds = Double(selectedStrength) * 0.4
        if seconds < 60 {
            return "\(Int(seconds))s"
        } else {
            let minutes = Int(seconds / 60)
            return "~\(minutes)m"
        }
    }
}
