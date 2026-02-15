//
//  ChatMessage.swift
//  Cena
//
//  Chat message model for the agent interface
//

import Foundation
import AppKit


// MARK: - Privacy Report Model

struct PrivacyEntityReport: Identifiable {
    let id = UUID()
    let piiType: String
    let action: String   // "KEEP", "BLUR", "REDACT"
    let token: String?
    let hint: String?

    var actionIcon: String {
        switch action {
        case "REDACT": return "eye.slash.fill"
        case "BLUR":   return "aqi.medium"
        case "KEEP":   return "eye.fill"
        default:       return "questionmark.circle"
        }
    }

    var actionColor: String {
        switch action {
        case "REDACT": return "red"
        case "BLUR":   return "yellow"
        case "KEEP":   return "green"
        default:       return "gray"
        }
    }

    /// Human-friendly PII type label
    var typeLabel: String {
        piiType
            .replacingOccurrences(of: "_", with: " ")
            .capitalized
    }
}

struct PrivacyReport {
    let totalDetected: Int
    let redactedCount: Int
    let blurredCount: Int
    let keptCount: Int
    let entities: [PrivacyEntityReport]
    let adversarialRisk: Double
    let privacyConfidence: Double
    let warnings: [String]
    let spacyActive: Bool

    var protectedCount: Int { redactedCount + blurredCount }

    /// Parse from the server's JSON dictionary
    static func from(_ dict: [String: Any]) -> PrivacyReport? {
        guard let total = dict["total_detected"] as? Int else { return nil }

        let entities: [PrivacyEntityReport] = (dict["entities"] as? [[String: Any]] ?? []).map { e in
            PrivacyEntityReport(
                piiType: e["pii_type"] as? String ?? "UNKNOWN",
                action: e["action"] as? String ?? "KEEP",
                token: e["token"] as? String,
                hint: e["hint"] as? String
            )
        }

        return PrivacyReport(
            totalDetected: total,
            redactedCount: dict["redacted_count"] as? Int ?? 0,
            blurredCount: dict["blurred_count"] as? Int ?? 0,
            keptCount: dict["kept_count"] as? Int ?? 0,
            entities: entities,
            adversarialRisk: dict["adversarial_risk"] as? Double ?? 0,
            privacyConfidence: dict["privacy_confidence"] as? Double ?? 1.0,
            warnings: dict["warnings"] as? [String] ?? [],
            spacyActive: dict["spacy_active"] as? Bool ?? false
        )
    }
}


// MARK: - Chat Message

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: Role
    let text: String
    let model: String?
    let image: NSImage?
    let sanitizedPrompt: String?
    let privacyReport: PrivacyReport?
    let timestamp = Date()

    enum Role {
        case user
        case assistant
        case error
        case status(String)
    }

    init(
        role: Role,
        text: String,
        model: String? = nil,
        image: NSImage? = nil,
        sanitizedPrompt: String? = nil,
        privacyReport: PrivacyReport? = nil
    ) {
        self.role = role
        self.text = text
        self.model = model
        self.image = image
        self.sanitizedPrompt = sanitizedPrompt
        self.privacyReport = privacyReport
    }

    var isUser: Bool {
        if case .user = role { return true }
        return false
    }

    var isStatus: Bool {
        if case .status = role { return true }
        return false
    }
}
