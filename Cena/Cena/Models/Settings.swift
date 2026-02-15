//
//  Settings.swift
//  Cena
//
//  User preferences and application settings
//

import Foundation

class Settings: ObservableObject, Codable {
    @Published var enabled: Bool = true
    @Published var selectedBackend: String = "gx10"
    @Published var monitorPasteboard: Bool = true
    @Published var autoApprove: Bool = false
    @Published var defaultIterations: Int = 200
    @Published var defaultMaskMode: String = "auto_face" // "auto_face", "full_image", "manual"
    @Published var showNotifications: Bool = true
    @Published var launchAtLogin: Bool = false

    // Configuration file path
    private static let configPath = FileManager.default
        .homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/Cena/config.json")

    enum CodingKeys: String, CodingKey {
        case enabled
        case selectedBackend = "backend"
        case monitorPasteboard = "monitor_pasteboard"
        case autoApprove = "auto_approve"
        case defaultIterations = "default_iterations"
        case defaultMaskMode = "mask_mode"
        case showNotifications = "show_notifications"
        case launchAtLogin = "launch_at_login"
    }

    init() {
        load()
    }

    /// Load settings from disk
    func load() {
        guard FileManager.default.fileExists(atPath: Settings.configPath.path) else {
            print("⚙️  No saved settings found, using defaults")
            return
        }

        do {
            let data = try Data(contentsOf: Settings.configPath)
            let decoder = JSONDecoder()
            let loaded = try decoder.decode(Settings.self, from: data)

            // Copy loaded values
            self.enabled = loaded.enabled
            self.selectedBackend = loaded.selectedBackend
            self.monitorPasteboard = loaded.monitorPasteboard
            self.autoApprove = loaded.autoApprove
            self.defaultIterations = loaded.defaultIterations
            self.defaultMaskMode = loaded.defaultMaskMode
            self.showNotifications = loaded.showNotifications
            self.launchAtLogin = loaded.launchAtLogin

            print("✅ Loaded settings from: \(Settings.configPath.path)")
        } catch {
            print("❌ Failed to load settings: \(error)")
        }
    }

    /// Save settings to disk
    func save() {
        do {
            // Create directory if needed
            let directory = Settings.configPath.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(self)
            try data.write(to: Settings.configPath)

            print("✅ Saved settings to: \(Settings.configPath.path)")
        } catch {
            print("❌ Failed to save settings: \(error)")
        }
    }

    // MARK: - Codable

    required init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        enabled = try container.decode(Bool.self, forKey: .enabled)
        selectedBackend = try container.decode(String.self, forKey: .selectedBackend)
        monitorPasteboard = try container.decode(Bool.self, forKey: .monitorPasteboard)
        autoApprove = try container.decode(Bool.self, forKey: .autoApprove)
        defaultIterations = try container.decode(Int.self, forKey: .defaultIterations)
        defaultMaskMode = try container.decode(String.self, forKey: .defaultMaskMode)
        showNotifications = try container.decode(Bool.self, forKey: .showNotifications)
        launchAtLogin = try container.decodeIfPresent(Bool.self, forKey: .launchAtLogin) ?? false
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(enabled, forKey: .enabled)
        try container.encode(selectedBackend, forKey: .selectedBackend)
        try container.encode(monitorPasteboard, forKey: .monitorPasteboard)
        try container.encode(autoApprove, forKey: .autoApprove)
        try container.encode(defaultIterations, forKey: .defaultIterations)
        try container.encode(defaultMaskMode, forKey: .defaultMaskMode)
        try container.encode(showNotifications, forKey: .showNotifications)
        try container.encode(launchAtLogin, forKey: .launchAtLogin)
    }

    /// Get mask mode enum from string
    func getMaskMode() -> GlazingJob.MaskMode {
        switch defaultMaskMode {
        case "auto_face":
            return .autoFace
        case "full_image":
            return .fullImage
        default:
            return .autoFace
        }
    }
}
