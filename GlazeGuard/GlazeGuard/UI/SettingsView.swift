//
//  SettingsView.swift
//  GlazeGuard
//
//  User settings and configuration panel
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

    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Image(systemName: "shield.checkered")
                    .font(.title2)
                    .foregroundStyle(
                        LinearGradient(
                            colors: [.blue, .purple],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )

                Text("GlazeGuard Settings")
                    .font(.title2.weight(.semibold))

                Spacer()

                Button(action: { dismiss() }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title3)
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
            }
            .padding()
            .background(Color(NSColor.windowBackgroundColor))

            Divider()

            // Settings content
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {

                    // General Section
                    SettingsSection(title: "General", icon: "gearshape") {
                        Toggle("Enable GlazeGuard", isOn: $settings.enabled)
                            .toggleStyle(.switch)

                        Toggle("Launch at Login", isOn: $settings.launchAtLogin)
                            .toggleStyle(.switch)

                        Toggle("Show Notifications", isOn: $settings.showNotifications)
                            .toggleStyle(.switch)
                    }

                    // Backend Section
                    SettingsSection(title: "Backend Server", icon: "server.rack") {
                        Picker("Backend:", selection: $selectedBackendName) {
                            ForEach(Array(backendConfig.backends.keys.sorted()), id: \.self) { key in
                                if let backend = backendConfig.backends[key] {
                                    Text("\(key.capitalized) - \(backend.description)")
                                        .tag(key)
                                }
                            }
                        }
                        .pickerStyle(.menu)
                        .onChange(of: selectedBackendName) { newValue in
                            settings.selectedBackend = newValue
                        }

                        if let backend = backendConfig.backends[selectedBackendName] {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("URL:")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                Text(backend.url)
                                    .font(.caption.monospaced())
                                    .foregroundColor(.blue)
                                    .textSelection(.enabled)
                            }
                            .padding(.top, 4)
                        }
                    }

                    // Protection Settings Section
                    SettingsSection(title: "Protection Settings", icon: "lock.shield") {
                        VStack(alignment: .leading, spacing: 12) {
                            HStack {
                                Text("Glazing Strength:")
                                    .frame(width: 140, alignment: .leading)

                                Picker("", selection: $settings.defaultIterations) {
                                    Text("Low (100 iters)").tag(100)
                                    Text("Medium (200 iters)").tag(200)
                                    Text("High (500 iters)").tag(500)
                                    Text("Maximum (1000 iters)").tag(1000)
                                }
                                .pickerStyle(.menu)
                            }

                            HStack {
                                Text("Protection Mode:")
                                    .frame(width: 140, alignment: .leading)

                                Picker("", selection: $settings.defaultMaskMode) {
                                    Text("Auto-detect Faces").tag("auto_face")
                                    Text("Full Image").tag("full_image")
                                }
                                .pickerStyle(.menu)
                            }

                            Text("Estimated time: \(estimatedTime)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }

                    // Pasteboard Section
                    SettingsSection(title: "Clipboard Monitoring", icon: "doc.on.clipboard") {
                        Toggle("Monitor Clipboard for Images", isOn: $settings.monitorPasteboard)
                            .toggleStyle(.switch)

                        Toggle("Auto-approve Protection", isOn: $settings.autoApprove)
                            .toggleStyle(.switch)
                            .disabled(!settings.monitorPasteboard)

                        Text("When enabled, GlazeGuard will automatically detect when you copy images and offer to protect them before pasting.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    Spacer()
                }
                .padding()
            }

            Divider()

            // Footer buttons
            HStack {
                Button("Reset to Defaults") {
                    resetToDefaults()
                }
                .buttonStyle(.borderless)

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
            .padding()
        }
        .frame(width: 600, height: 550)
    }

    // MARK: - Helpers

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

// MARK: - Settings Section Component

struct SettingsSection<Content: View>: View {
    let title: String
    let icon: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .foregroundColor(.blue)
                Text(title)
                    .font(.headline)
            }

            VStack(alignment: .leading, spacing: 12) {
                content
            }
            .padding(.leading, 28)
        }
    }
}

// MARK: - Preview

#if DEBUG
struct SettingsView_Previews: PreviewProvider {
    static var previews: some View {
        let settings = Settings()
        let config = BackendConfig.load() ?? BackendConfig(
            backends: [
                "local": BackendConfig.Backend(
                    url: "http://localhost:8888",
                    description: "Local machine",
                    type: "local",
                    ssh: nil,
                    podId: nil,
                    apiKey: nil
                )
            ],
            defaultBackend: "local"
        )

        SettingsView(settings: settings, backendConfig: config)
    }
}
#endif
