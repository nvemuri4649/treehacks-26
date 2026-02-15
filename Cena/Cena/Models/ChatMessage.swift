//
//  ChatMessage.swift
//  Cena
//
//  Chat message model for the agent interface
//

import Foundation
import AppKit

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: Role
    let text: String
    let model: String?
    let image: NSImage?
    let timestamp = Date()

    enum Role {
        case user
        case assistant
        case error
        case status(String)
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
