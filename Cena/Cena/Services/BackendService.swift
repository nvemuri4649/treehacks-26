//
//  BackendService.swift
//  Cena
//
//  HTTP client for communicating with the encryption server API
//

import Foundation
import AppKit

struct HealthResponse: Codable {
    let status: String
    let gpu: GPUInfo?
    let gpu_memory: String?
    let device: String?
    let model_loaded: Bool?

    struct GPUInfo: Codable {
        let name: String?
        let memory: String?
    }

    // Handle "gpu" being either a string ("NVIDIA GB10") or an object ({name, memory})
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        status = try container.decode(String.self, forKey: .status)
        gpu_memory = try container.decodeIfPresent(String.self, forKey: .gpu_memory)
        device = try container.decodeIfPresent(String.self, forKey: .device)
        model_loaded = try container.decodeIfPresent(Bool.self, forKey: .model_loaded)

        // Try decoding gpu as GPUInfo object first, then as plain string
        if let gpuObj = try? container.decodeIfPresent(GPUInfo.self, forKey: .gpu) {
            gpu = gpuObj
        } else if let gpuString = try? container.decodeIfPresent(String.self, forKey: .gpu) {
            gpu = GPUInfo(name: gpuString, memory: gpu_memory)
        } else {
            gpu = nil
        }
    }
}

class BackendService {
    private let baseURL: String
    private let session: URLSession

    init(baseURL: String) {
        self.baseURL = baseURL

        // Configure session with longer timeout for glazing operations
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30  // Health check timeout
        config.timeoutIntervalForResource = 900  // 15 minutes for glazing
        self.session = URLSession(configuration: config)
    }

    // MARK: - Health Check

    /// Check if backend server is available and get GPU info
    func checkHealth() async throws -> HealthResponse {
        guard let url = URL(string: "\(baseURL)/health") else {
            throw GlazingError.backendUnavailable
        }

        do {
            let (data, response) = try await session.data(from: url)

            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                throw GlazingError.backendUnavailable
            }

            let healthResponse = try JSONDecoder().decode(HealthResponse.self, from: data)
            print("✅ Backend health check passed: \(healthResponse.status)")
            if let gpu = healthResponse.gpu {
                print("   GPU: \(gpu.name ?? "unknown") - \(gpu.memory ?? "unknown")")
            }
            return healthResponse
        } catch {
            print("❌ Health check failed: \(error)")
            throw GlazingError.networkError(error)
        }
    }

    // MARK: - Image Protection

    /// Protect an image using the /protect endpoint
    /// - Parameters:
    ///   - image: Original image to protect
    ///   - mask: Binary mask (white = protect, black = leave alone)
    ///   - iterations: Number of adversarial iterations (100-1000)
    ///   - progressCallback: Called periodically with progress updates (0.0 to 1.0)
    /// - Returns: Protected image with adversarial perturbations
    func protectImage(
        image: NSImage,
        mask: NSImage,
        iterations: Int = 200,
        progressCallback: ((Double) -> Void)? = nil
    ) async throws -> NSImage {
        guard let url = URL(string: "\(baseURL)/protect?iters=\(iterations)") else {
            throw GlazingError.backendUnavailable
        }

        // Convert images to PNG data
        guard let imageData = image.pngData(),
              let maskData = mask.pngData() else {
            throw GlazingError.invalidImage
        }

        // Build multipart/form-data request
        var request = URLRequest(url: url)
        request.httpMethod = "POST"

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        // Create multipart body
        var body = Data()

        // Add image file
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"image\"; filename=\"image.png\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/png\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n".data(using: .utf8)!)

        // Add mask file
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"mask\"; filename=\"mask.png\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/png\r\n\r\n".data(using: .utf8)!)
        body.append(maskData)
        body.append("\r\n".data(using: .utf8)!)

        // Close boundary
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)

        request.httpBody = body
        request.timeoutInterval = 900  // 15 minutes

        print("🔐 Protecting image: \(iterations) iterations, size: \(image.size)")

        // Start request
        let startTime = Date()
        do {
            let (data, response) = try await session.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                throw GlazingError.invalidResponse
            }

            if httpResponse.statusCode != 200 {
                print("❌ Server returned error: \(httpResponse.statusCode)")
                throw GlazingError.invalidResponse
            }

            // Parse response as image
            guard let protectedImage = NSImage(data: data) else {
                print("❌ Server returned invalid image data")
                throw GlazingError.invalidResponse
            }

            let elapsed = Date().timeIntervalSince(startTime)
            print("✅ Image protected successfully in \(Int(elapsed))s")

            return protectedImage

        } catch let error as GlazingError {
            throw error
        } catch {
            print("❌ Network error during protection: \(error)")
            throw GlazingError.networkError(error)
        }
    }

    // MARK: - Helper Methods

    /// Test if backend is reachable
    func isReachable() async -> Bool {
        do {
            _ = try await checkHealth()
            return true
        } catch {
            return false
        }
    }
}

// MARK: - NSImage Extensions

extension NSImage {
    /// Convert NSImage to PNG data
    func pngData() -> Data? {
        guard let cgImage = self.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return nil
        }

        let rep = NSBitmapImageRep(cgImage: cgImage)
        rep.size = self.size  // Match original size

        return rep.representation(using: .png, properties: [:])
    }

    /// Convert NSImage to JPEG data
    func jpegData(compressionQuality: Double = 0.9) -> Data? {
        guard let cgImage = self.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return nil
        }

        let rep = NSBitmapImageRep(cgImage: cgImage)
        rep.size = self.size

        return rep.representation(using: .jpeg, properties: [.compressionFactor: compressionQuality])
    }

    func blendWith(original: NSImage, alpha: Double) -> NSImage {
        guard let protCG = self.cgImage(forProposedRect: nil, context: nil, hints: nil),
              let origCG = original.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return self }
        let w = protCG.width, h = protCG.height
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: w * 4,
            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return self }

        ctx.draw(origCG, in: CGRect(x: 0, y: 0, width: w, height: h))
        ctx.setAlpha(CGFloat(alpha))
        ctx.draw(protCG, in: CGRect(x: 0, y: 0, width: w, height: h))

        guard let out = ctx.makeImage() else { return self }
        return NSImage(cgImage: out, size: self.size)
    }

    /// Optional finalization pass (e.g. for export). No visible markers applied.
    func applyFinalization() -> NSImage {
        return self
    }
}
