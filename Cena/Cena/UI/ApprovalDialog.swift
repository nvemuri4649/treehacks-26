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
    @State private var rememberChoice = false

    enum ApprovalOption {
        case once(iterations: Int, maskMode: String)
        case always(iterations: Int, maskMode: String)
    }

    var body: some View {
        VStack(spacing: 20) {
            // Header with icon
            HStack {
                Image(systemName: "shield.lefthalf.filled")
                    .font(.system(size: 48))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [.blue, .purple],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )

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

            // Image preview
            HStack {
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
                    // Strength picker
                    HStack {
                        Text("Strength:")
                            .frame(width: 80, alignment: .leading)

                        Picker("", selection: $selectedStrength) {
                            Text("Low (100)").tag(100)
                            Text("Medium (200)").tag(200)
                            Text("High (500)").tag(500)
                        }
                        .pickerStyle(.segmented)
                    }

                    // Mask mode picker
                    HStack {
                        Text("Region:")
                            .frame(width: 80, alignment: .leading)

                        Picker("", selection: $selectedMaskMode) {
                            Text("Faces").tag("auto_face")
                            Text("Full Image").tag("full_image")
                        }
                        .pickerStyle(.segmented)
                    }

                    // Time estimate
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

            // Remember choice checkbox
            Toggle("Always encrypt copied images", isOn: $rememberChoice)
                .toggleStyle(.checkbox)
                .font(.caption)

            Divider()

            // Action buttons
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
                        option = .always(iterations: selectedStrength, maskMode: selectedMaskMode)
                    } else {
                        option = .once(iterations: selectedStrength, maskMode: selectedMaskMode)
                    }
                    onApprove(option)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(24)
        .frame(width: 500)
        .background(
            VisualEffectViewRepresentable(material: .hudWindow, blendingMode: .behindWindow)
                .ignoresSafeArea()
        )
    }

    // MARK: - Helpers

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

// MARK: - Preview

#if DEBUG
struct ApprovalDialog_Previews: PreviewProvider {
    static var previews: some View {
        ApprovalDialog(
            image: NSImage(systemSymbolName: "photo", accessibilityDescription: nil) ?? NSImage(),
            onApprove: { _ in print("Approved") },
            onDeny: { print("Denied") }
        )
    }
}
#endif
