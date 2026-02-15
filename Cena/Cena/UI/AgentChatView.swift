//
//  AgentChatView.swift
//  Cena
//
//  Native SwiftUI chat interface for the Cena privacy agent
//

import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct AgentChatView: View {
    @StateObject private var ws = AgentWebSocket()
    @State private var messages: [ChatMessage] = []
    @State private var inputText = ""
    @State private var selectedModel = "claude-sonnet-4-20250514"
    @State private var isProcessing = false
    @State private var pendingImage: NSImage?
    @State private var pendingImageData: Data?
    @State private var pendingMimeType: String?
    @State private var sessionId: String?
    @State private var showImagePicker = false

    private let models = [
        ("Claude Sonnet 4", "claude-sonnet-4-20250514"),
        ("Claude Haiku 4", "claude-haiku-4-20250414"),
        ("GPT-4o", "gpt-4o"),
        ("GPT-4o Mini", "gpt-4o-mini"),
    ]

    var body: some View {
        ZStack {
            // Background
            VisualEffectViewRepresentable(
                material: .sidebar,
                blendingMode: .behindWindow
            )
            .ignoresSafeArea()

            VStack(spacing: 0) {
                headerBar
                Divider().opacity(0.3)
                chatArea
                statusIndicator
                imagePreviewBar
                Divider().opacity(0.3)
                inputBar
            }
        }
        .frame(minWidth: 580, minHeight: 500)
        .onAppear { initSession() }
        .onDisappear { ws.disconnect() }
    }

    // MARK: - Header

    private var headerBar: some View {
        HStack(spacing: 12) {
            Image(systemName: "shield.checkered")
                .font(.title2)
                .foregroundStyle(
                    LinearGradient(
                        colors: [.blue, .purple],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )

            VStack(alignment: .leading, spacing: 1) {
                Text("Cena Agent")
                    .font(.headline)
                Text("Privacy-first cloud reasoning")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }

            Spacer()

            Picker("", selection: $selectedModel) {
                ForEach(models, id: \.1) { name, id in
                    Text(name).tag(id)
                }
            }
            .pickerStyle(.menu)
            .frame(width: 180)

            Button {
                initSession()
            } label: {
                Image(systemName: "plus.circle")
                    .font(.title3)
                    .foregroundColor(.secondary)
            }
            .buttonStyle(.plain)
            .help("New session")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    // MARK: - Chat area

    private var chatArea: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 12) {
                    if messages.isEmpty {
                        welcomeView
                    }

                    ForEach(messages) { msg in
                        messageBubble(msg)
                            .id(msg.id)
                    }
                }
                .padding(16)
            }
            .onChange(of: messages.count) { _, _ in
                if let last = messages.last {
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
    }

    private var welcomeView: some View {
        VStack(spacing: 16) {
            Spacer()

            Image(systemName: "shield.checkered")
                .font(.system(size: 48))
                .foregroundStyle(
                    LinearGradient(
                        colors: [.blue.opacity(0.5), .purple.opacity(0.5)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )

            Text("Your conversations, protected.")
                .font(.title3.weight(.semibold))

            Text("Personal information is dereferenced and likenesses are encrypted locally before reaching any cloud model.")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 380)

            HStack(spacing: 8) {
                featurePill("PII Dereferencing")
                featurePill("Likeness Encryption")
                featurePill("Local Processing")
            }

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(.vertical, 40)
    }

    private func featurePill(_ text: String) -> some View {
        Text(text)
            .font(.caption2.weight(.medium))
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.purple.opacity(0.15))
            )
            .foregroundColor(.purple)
    }

    // MARK: - Message bubble

    @ViewBuilder
    private func messageBubble(_ msg: ChatMessage) -> some View {
        switch msg.role {
        case .user:
            HStack {
                Spacer(minLength: 80)
                VStack(alignment: .trailing, spacing: 4) {
                    if let img = msg.image {
                        Image(nsImage: img)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: 200, maxHeight: 140)
                            .cornerRadius(10)
                    }
                    Text(msg.text)
                        .padding(12)
                        .background(
                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                .fill(LinearGradient(
                                    colors: [Color(nsColor: .controlAccentColor), .purple],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ))
                        )
                        .foregroundColor(.white)
                }
            }

        case .assistant:
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(msg.text)
                        .padding(12)
                        .background(
                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                .fill(Color(nsColor: .controlBackgroundColor).opacity(0.6))
                        )
                        .textSelection(.enabled)

                    if let model = msg.model {
                        Text(model)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .padding(.leading, 4)
                    }
                }
                Spacer(minLength: 80)
            }

        case .error:
            HStack {
                VStack(alignment: .leading) {
                    Text(msg.text)
                        .padding(12)
                        .background(
                            RoundedRectangle(cornerRadius: 16, style: .continuous)
                                .fill(Color.red.opacity(0.12))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                                        .strokeBorder(Color.red.opacity(0.25), lineWidth: 1)
                                )
                        )
                        .foregroundColor(.red)
                }
                Spacer(minLength: 80)
            }

        case .status:
            EmptyView()
        }
    }

    // MARK: - Status indicator

    @ViewBuilder
    private var statusIndicator: some View {
        if let stage = ws.currentStage {
            HStack(spacing: 8) {
                Circle()
                    .fill(stageColor(stage))
                    .frame(width: 8, height: 8)
                    .modifier(PulseAnimation())

                Text(stageLabel(stage))
                    .font(.caption)
                    .foregroundColor(.secondary)

                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 6)
            .transition(.opacity)
        }
    }

    private func stageColor(_ stage: String) -> Color {
        switch stage {
        case "sanitizing": return .yellow
        case "glazing": return .purple
        case "thinking": return .blue
        case "restoring": return .green
        default: return .gray
        }
    }

    private func stageLabel(_ stage: String) -> String {
        switch stage {
        case "sanitizing": return "Dereferencing personal information locally..."
        case "glazing": return "Encrypting likeness in uploaded image..."
        case "thinking": return "Cloud model is thinking (sanitized data only)..."
        case "restoring": return "Re-referencing your information locally..."
        default: return stage
        }
    }

    // MARK: - Image preview

    @ViewBuilder
    private var imagePreviewBar: some View {
        if let img = pendingImage {
            HStack(spacing: 8) {
                Image(nsImage: img)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 48, height: 48)
                    .cornerRadius(8)
                    .clipped()

                Text("Image attached")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Spacer()

                Button {
                    clearImage()
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 6)
        }
    }

    // MARK: - Input bar

    private var inputBar: some View {
        HStack(alignment: .bottom, spacing: 8) {
            Button {
                pickImage()
            } label: {
                Image(systemName: "photo.badge.plus")
                    .font(.title3)
                    .foregroundColor(.secondary)
            }
            .buttonStyle(.plain)
            .help("Upload image")

            ZStack(alignment: .leading) {
                if inputText.isEmpty {
                    Text("Type your message...")
                        .foregroundColor(.secondary.opacity(0.6))
                        .padding(.leading, 12)
                        .padding(.vertical, 10)
                }

                TextEditor(text: $inputText)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .frame(minHeight: 36, maxHeight: 120)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color(nsColor: .controlBackgroundColor).opacity(0.4))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .strokeBorder(Color(nsColor: .separatorColor).opacity(0.3), lineWidth: 1)
                    )
            )

            Button {
                sendMessage()
            } label: {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.title2)
                    .foregroundStyle(sendButtonDisabled ? AnyShapeStyle(Color.gray) : AnyShapeStyle(
                        LinearGradient(
                            colors: [.blue, .purple],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    ))
            }
            .buttonStyle(.plain)
            .disabled(sendButtonDisabled)
            .help("Send message")
            .keyboardShortcut(.return, modifiers: [])
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var sendButtonDisabled: Bool {
        isProcessing || inputText.trimmingCharacters(in: .whitespaces).isEmpty
    }

    // MARK: - Actions

    private func initSession() {
        messages = []
        ws.disconnect()
        isProcessing = false
        clearImage()

        Task {
            do {
                let sid = try await ws.createSession()
                sessionId = sid
                ws.connect(sessionId: sid)

                ws.onMessage = { msg in
                    handleServerMessage(msg)
                }

                print("🔗 Agent session: \(sid)")
            } catch {
                messages.append(ChatMessage(
                    role: .error,
                    text: "Cannot connect to agent server at 127.0.0.1:8000. Is it running?",
                    model: nil,
                    image: nil
                ))
            }
        }
    }

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty, !isProcessing else { return }

        let userMsg = ChatMessage(
            role: .user,
            text: text,
            model: nil,
            image: pendingImage
        )
        messages.append(userMsg)

        ws.send(
            text: text,
            model: selectedModel,
            imageData: pendingImageData,
            mimeType: pendingMimeType
        )

        inputText = ""
        clearImage()
        isProcessing = true
    }

    private func handleServerMessage(_ msg: AgentWebSocket.ServerMessage) {
        switch msg.type {
        case .status:
            break // handled by ws.currentStage binding

        case .response(let text, let model, _, _):
            messages.append(ChatMessage(
                role: .assistant,
                text: text,
                model: model,
                image: nil
            ))
            isProcessing = false

        case .error(let message):
            messages.append(ChatMessage(
                role: .error,
                text: message,
                model: nil,
                image: nil
            ))
            isProcessing = false
        }
    }

    private func pickImage() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.png, .jpeg, .gif, .webP]
        panel.message = "Select an image to send"

        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }

            Task { @MainActor in
                if let data = try? Data(contentsOf: url),
                   let image = NSImage(data: data) {
                    pendingImage = image
                    pendingImageData = data

                    let ext = url.pathExtension.lowercased()
                    switch ext {
                    case "png": pendingMimeType = "image/png"
                    case "jpg", "jpeg": pendingMimeType = "image/jpeg"
                    case "gif": pendingMimeType = "image/gif"
                    case "webp": pendingMimeType = "image/webp"
                    default: pendingMimeType = "image/png"
                    }
                }
            }
        }
    }

    private func clearImage() {
        pendingImage = nil
        pendingImageData = nil
        pendingMimeType = nil
    }
}

// MARK: - Pulse animation modifier

struct PulseAnimation: ViewModifier {
    @State private var opacity: Double = 1.0

    func body(content: Content) -> some View {
        content
            .opacity(opacity)
            .onAppear {
                withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) {
                    opacity = 0.3
                }
            }
    }
}
