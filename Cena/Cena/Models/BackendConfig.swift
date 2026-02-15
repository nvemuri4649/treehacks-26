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
        var configPath: String

        if path.isEmpty {
            // Search upward from the binary to find backends.json in the project root
            let searchPaths = [
                // Relative to binary (e.g. Cena/.build/release/ -> project root)
                URL(fileURLWithPath: CommandLine.arguments[0])
                    .deletingLastPathComponent()
                    .deletingLastPathComponent()
                    .deletingLastPathComponent()
                    .deletingLastPathComponent()
                    .appendingPathComponent("backends.json").path,
                // Common project paths
                "/Users/nikhil/Documents/StudioProjects/treehacks-26/backends.json",
                // Fallback: current working directory
                FileManager.default.currentDirectoryPath + "/backends.json",
            ]

            var found = false
            configPath = ""
            for p in searchPaths {
                if FileManager.default.fileExists(atPath: p) {
                    configPath = p
                    found = true
                    break
                }
            }

            if !found {
                // Try bundle as last resort
                guard let bundlePath = Bundle.main.path(forResource: "backends", ofType: "json") else {
                    print("❌ backends.json not found. Searched: \(searchPaths)")
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
