//
//  AgentWebSocket.swift
//  Cena
//
//  WebSocket client for the Cena privacy agent backend
//

import Foundation

@MainActor
class AgentWebSocket: ObservableObject {
    @Published var isConnected = false
    @Published var currentStage: String?

    private var webSocket: URLSessionWebSocketTask?
    private var session: URLSession?
    private var serverURL: String

    var onMessage: ((ServerMessage) -> Void)?

    // MARK: - Server message types

    struct ServerMessage {
        enum MessageType {
            case status(stage: String)
            case response(text: String, model: String?, hadImage: Bool, sanitizedPrompt: String?)
            case error(message: String)
        }
        let type: MessageType
    }

    init(serverURL: String = "http://127.0.0.1:8000") {
        self.serverURL = serverURL
    }

    // MARK: - Session management

    func createSession() async throws -> String {
        let url = URL(string: "\(serverURL)/api/session")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"

        let (data, _) = try await URLSession.shared.data(for: request)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let sessionId = json?["session_id"] as? String else {
            throw AgentError.invalidResponse
        }
        return sessionId
    }

    func connect(sessionId: String) {
        let wsURL = serverURL
            .replacingOccurrences(of: "http://", with: "ws://")
            .replacingOccurrences(of: "https://", with: "wss://")

        guard let url = URL(string: "\(wsURL)/ws/\(sessionId)") else { return }

        session = URLSession(configuration: .default)
        webSocket = session?.webSocketTask(with: url)
        webSocket?.resume()
        isConnected = true
        receiveMessage()
    }

    func disconnect() {
        webSocket?.cancel(with: .goingAway, reason: nil)
        webSocket = nil
        isConnected = false
        currentStage = nil
    }

    // MARK: - Send

    func send(text: String, model: String, imageData: Data? = nil, mimeType: String? = nil) {
        var payload: [String: Any] = [
            "text": text,
            "model": model,
        ]

        if let imageData = imageData, let mimeType = mimeType {
            payload["image"] = imageData.base64EncodedString()
            payload["mime_type"] = mimeType
        }

        guard let data = try? JSONSerialization.data(withJSONObject: payload),
              let jsonString = String(data: data, encoding: .utf8) else { return }

        webSocket?.send(.string(jsonString)) { error in
            if let error = error {
                print("❌ WebSocket send error: \(error)")
            }
        }
    }

    // MARK: - Receive

    private func receiveMessage() {
        webSocket?.receive { [weak self] result in
            Task { @MainActor in
                switch result {
                case .success(let message):
                    switch message {
                    case .string(let text):
                        self?.handleRawMessage(text)
                    default:
                        break
                    }
                    self?.receiveMessage()

                case .failure(let error):
                    print("❌ WebSocket receive error: \(error)")
                    self?.isConnected = false
                }
            }
        }
    }

    private func handleRawMessage(_ text: String) {
        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = json["type"] as? String else { return }

        switch type {
        case "status":
            let stage = json["stage"] as? String ?? ""
            currentStage = stage
            onMessage?(ServerMessage(type: .status(stage: stage)))

        case "response":
            currentStage = nil
            let responseText = json["text"] as? String ?? ""
            let model = json["model"] as? String
            let hadImage = json["had_image"] as? Bool ?? false
            let sanitized = json["sanitized_prompt"] as? String
            onMessage?(ServerMessage(type: .response(
                text: responseText,
                model: model,
                hadImage: hadImage,
                sanitizedPrompt: sanitized
            )))

        case "error":
            currentStage = nil
            let message = json["message"] as? String ?? "Unknown error"
            onMessage?(ServerMessage(type: .error(message: message)))

        default:
            break
        }
    }

    enum AgentError: LocalizedError {
        case invalidResponse
        case connectionFailed

        var errorDescription: String? {
            switch self {
            case .invalidResponse: return "Invalid server response"
            case .connectionFailed: return "Failed to connect to agent server"
            }
        }
    }
}
