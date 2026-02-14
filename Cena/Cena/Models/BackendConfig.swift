//
//  BackendConfig.swift
//  Cena
//
//  Model for parsing backends.json configuration file
//

import Foundation

struct BackendConfig: Codable {
    let backends: [String: Backend]
    let defaultBackend: String

    enum CodingKeys: String, CodingKey {
        case backends
        case defaultBackend = "default"
    }

    struct Backend: Codable {
        let url: String
        let description: String
        let type: String
        let ssh: String?
        let podId: String?
        let apiKey: String?

        enum CodingKeys: String, CodingKey {
            case url
            case description
            case type
            case ssh
            case podId = "pod_id"
            case apiKey = "api_key"
        }
    }

    /// Load backend configuration from backends.json in the project directory
    static func load(from path: String = "") -> BackendConfig? {
        let configPath: String

        if path.isEmpty {
            // Try project directory first
            let projectPath = "/Users/shahabhishek1729/Documents/Stanford/Notes/EC/TreeHacks26/treehacks-26/backends.json"
            if FileManager.default.fileExists(atPath: projectPath) {
                configPath = projectPath
            } else {
                // Fallback to bundle
                guard let bundlePath = Bundle.main.path(forResource: "backends", ofType: "json") else {
                    print("❌ backends.json not found in project or bundle")
                    return nil
                }
                configPath = bundlePath
            }
        } else {
            configPath = path
        }

        do {
            let data = try Data(contentsOf: URL(fileURLWithPath: configPath))
            let config = try JSONDecoder().decode(BackendConfig.self, from: data)
            print("✅ Loaded backend config from: \(configPath)")
            print("   Default backend: \(config.defaultBackend)")
            print("   Available backends: \(config.backends.keys.sorted().joined(separator: ", "))")
            return config
        } catch {
            print("❌ Failed to load backends.json: \(error)")
            return nil
        }
    }

    /// Get the default backend
    func getDefaultBackend() -> Backend? {
        return backends[defaultBackend]
    }

    /// Get a specific backend by name
    func getBackend(named name: String) -> Backend? {
        return backends[name]
    }
}
