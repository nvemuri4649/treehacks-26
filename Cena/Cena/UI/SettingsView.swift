//
// SettingsView.swift
// Cena
//
// User settings and configuration panel — glass aesthetic
//

import SwiftUI

struct SettingsView: View {
    @ObservedObject var settings: Settings
    @Environment(\.dismiss) var dismiss

    let backendConfig: BackendConfig
    @State private var selectedBackendName: String

    init(settings: Settings, backendConfig: BackendConfig) {
        self.settings = settings
        self.backendConfig = backendConfig
        self._selectedBackendName = State(initialValue: settings.selectedBackend)
    }

    private let accentGrad = LinearGradient(
        colors: [.blue, .purple],
        startPoint: .topLeading, endPoint: .bottomTrailing
    )

    var body: some View {
        VStack(spacing: 0) {
            //Header
            HStack(spacing: 10) {
                CenaLogo(size: 22, isAnimating: false, color: .white.opacity(0.8))

                Text("Settings")
                    .font(.system(size: 15, weight: .semibold, design: .rounded))

                Spacer()

                Button(action: { dismiss() }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 16))
                        .foregroundColor(.white.opacity(0.25))
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
            .padding(.top, 12)

            Rectangle().fill(.white.opacity(0.06)).frame(height: 0.5)

            //Content
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {

                    SettingsSection(title: "General", icon: "gearshape") {
                        SettingsToggle(label: "Enable Cena", isOn: $settings.enabled)
                        SettingsToggle(label: "Launch at Login", isOn: $settings.launchAtLogin)
                        SettingsToggle(label: "Show Notifications", isOn: $settings.showNotifications)
                    }

                    SettingsSection(title: "Backend Server", icon: "server.rack") {
                        Picker("Backend:", selection: $selectedBackendName) {
                            ForEach(Array(backendConfig.backends.keys.sorted()), id: \.self) { key in
                                if let backend = backendConfig.backends[key] {
                                    Text("\(key.capitalized) — \(backend.description)")
                                        .tag(key)
                                }
                            }
                        }
                        .pickerStyle(.menu)
                        .onChange(of: selectedBackendName) { _, newValue in
                            settings.selectedBackend = newValue
                        }

                        if let backend = backendConfig.backends[selectedBackendName] {
                            Text(backend.url)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundColor(.blue.opacity(0.7))
                                .textSelection(.enabled)
                                .padding(.top, 2)
                        }
                    }

                    SettingsSection(title: "Encryption", icon: "lock.shield") {
                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                Text("Strength:")
                                    .font(.system(size: 12))
                                    .foregroundColor(.white.opacity(0.5))
                                    .frame(width: 80, alignment: .leading)

                                Picker("", selection: $settings.defaultIterations) {
                                    Text("50").tag(50)
                                    Text("100").tag(100)
                                    Text("200").tag(200)
                                    Text("500").tag(500)
                                    Text("1000").tag(1000)
                                }
                                .pickerStyle(.segmented)
                            }

                            HStack {
                                Text("Region:")
                                    .font(.system(size: 12))
                                    .foregroundColor(.white.opacity(0.5))
                                    .frame(width: 80, alignment: .leading)

                                Picker("", selection: $settings.defaultMaskMode) {
                                    Text("Faces").tag("auto_face")
                                    Text("Full Image").tag("full_image")
                                }
                                .pickerStyle(.segmented)
                            }

                            HStack(spacing: 4) {
                                Image(systemName: "clock")
                                    .font(.system(size: 9))
                                Text("Est. time: \(estimatedTime)")
                                    .font(.system(size: 10))
                            }
                            .foregroundColor(.white.opacity(0.3))
                        }
                    }

                    SettingsSection(title: "Clipboard", icon: "doc.on.clipboard") {
                        SettingsToggle(label: "Monitor clipboard for images", isOn: $settings.monitorPasteboard)

                        SettingsToggle(label: "Auto-approve encryption", isOn: $settings.autoApprove)
                            .disabled(!settings.monitorPasteboard)

                        Text("When enabled, Cena automatically detects copied images and encrypts your likeness before pasting.")
                            .font(.system(size: 10))
                            .foregroundColor(.white.opacity(0.25))
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    Spacer(minLength: 10)
                }
                .padding(20)
            }

            Rectangle().fill(.white.opacity(0.06)).frame(height: 0.5)

            //Footer
            HStack {
                Button("Reset Defaults") {
                    resetToDefaults()
                }
                .font(.system(size: 11))
                .buttonStyle(.borderless)
                .foregroundColor(.white.opacity(0.4))

                Spacer()

                Button("Cancel") {
                    dismiss()
                }
                .keyboardShortcut(.cancelAction)

                Button("Save") {
                    saveSettings()
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
        }
        .frame(width: 520, height: 540)
        .background(
            VisualEffectViewRepresentable(material: .hudWindow, blendingMode: .behindWindow)
                .ignoresSafeArea()
        )
    }

    //MARK: - Helpers

    private var estimatedTime: String {
        let seconds = Double(settings.defaultIterations) * 0.4
        if seconds < 60 {
            return "\(Int(seconds))s"
        } else {
            let minutes = Int(seconds / 60)
            return "~\(minutes)m"
        }
    }

    private func saveSettings() {
        settings.save()
        print("💾 Settings saved")
    }

    private func resetToDefaults() {
        settings.enabled = true
        settings.selectedBackend = backendConfig.defaultBackend
        settings.monitorPasteboard = true
        settings.autoApprove = false
        settings.defaultIterations = 200
        settings.defaultMaskMode = "auto_face"
        settings.showNotifications = true
        settings.launchAtLogin = false
        print("🔄 Settings reset to defaults")
    }
}

//MARK: - Section

struct SettingsSection<Content: View>: View {
    let title: String
    let icon: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(
                        LinearGradient(colors: [.blue, .purple],
                                       startPoint: .topLeading, endPoint: .bottomTrailing)
                    )
                Text(title)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundColor(.white.opacity(0.7))
            }

            VStack(alignment: .leading, spacing: 10) {
                content
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(.white.opacity(0.06), lineWidth: 0.5)
            )
        }
    }
}

//MARK: - Toggle row

struct SettingsToggle: View {
    let label: String
    @Binding var isOn: Bool

    var body: some View {
        Toggle(label, isOn: $isOn)
            .toggleStyle(.switch)
            .font(.system(size: 12))
    }
}
