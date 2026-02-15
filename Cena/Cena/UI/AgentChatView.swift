//
//  AgentChatView.swift
//  Cena
//
//  Minimal translucent chat interface
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

    private let models: [(icon: String, name: String, id: String)] = [
        ("sparkle", "Claude Sonnet 4", "claude-sonnet-4-20250514"),
        ("hare", "Claude Haiku 4", "claude-haiku-4-20250414"),
        ("brain", "GPT-4o", "gpt-4o"),
        ("brain.filled.head.profile", "GPT-4o Mini", "gpt-4o-mini"),
    ]

    var body: some View {
        VStack(spacing: 0) {
            headerBar
            sep
            chatArea
            statusIndicator
            imagePreviewBar
            sep
            inputBar
        }
        .frame(minWidth: 620, minHeight: 480)
        .background(.clear)
        .onAppear { initSession() }
        .onDisappear { ws.disconnect() }
    }

    private var sep: some View {
        Rectangle().fill(.white.opacity(0.05)).frame(height: 0.5)
    }

    // MARK: - Header

    private var headerBar: some View {
        HStack(spacing: 10) {
            Text("Cena")
                .font(.system(size: 14, weight: .semibold, design: .rounded))
                .foregroundStyle(.primary.opacity(0.85))

            Spacer()

            modelSelector

            Button { initSession() } label: {
                Image(systemName: "arrow.counterclockwise")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.tertiary)
                    .padding(5)
                    .background(.white.opacity(0.05), in: Circle())
            }
            .buttonStyle(.plain)
            .help("New session")
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
        .padding(.top, 16)
    }

    private var selectedModelName: String {
        models.first(where: { $0.id == selectedModel })?.name ?? "Model"
    }

    private var modelSelector: some View {
        Menu {
            ForEach(models, id: \.id) { m in
                Button { selectedModel = m.id } label: {
                    Label(m.name, systemImage: m.icon)
                }
            }
        } label: {
            HStack(spacing: 5) {
                Text(selectedModelName)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.primary.opacity(0.6))
                Image(systemName: "chevron.down")
                    .font(.system(size: 7, weight: .bold))
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(.white.opacity(0.05), in: Capsule())
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
    }

    // MARK: - Chat

    private var chatArea: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 14) {
                    if messages.isEmpty {
                        welcomeView
                    }
                    ForEach(messages) { msg in
                        messageBubble(msg).id(msg.id)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
            }
            .onChange(of: messages.count) { _, _ in
                if let last = messages.last {
                    withAnimation(.easeOut(duration: 0.15)) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
    }

    private var welcomeView: some View {
        VStack(spacing: 10) {
            Spacer(minLength: 80)
            Text("What can I help with?")
                .font(.system(size: 20, weight: .medium, design: .rounded))
                .foregroundStyle(.primary.opacity(0.5))
            Text("All data is processed locally before reaching the cloud.")
                .font(.system(size: 11))
                .foregroundStyle(.tertiary)
            Spacer(minLength: 80)
        }
    }

    // MARK: - Bubbles

    @ViewBuilder
    private func messageBubble(_ msg: ChatMessage) -> some View {
        switch msg.role {
        case .user:
            HStack {
                Spacer(minLength: 80)
                Text(msg.text)
                    .font(.system(size: 13))
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(
                        .linearGradient(colors: [.blue.opacity(0.55), .purple.opacity(0.5)],
                                        startPoint: .topLeading, endPoint: .bottomTrailing),
                        in: RoundedRectangle(cornerRadius: 18, style: .continuous)
                    )
                    .foregroundColor(.white)
            }

        case .assistant:
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text(msg.text)
                        .font(.system(size: 13))
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(.white.opacity(0.06),
                                    in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                        .textSelection(.enabled)

                    if let model = msg.model {
                        Text(model)
                            .font(.system(size: 9))
                            .foregroundStyle(.quaternary)
                            .padding(.leading, 6)
                    }
                }
                Spacer(minLength: 80)
            }

        case .error:
            HStack {
                Text(msg.text)
                    .font(.system(size: 12))
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(Color.red.opacity(0.08),
                                in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .foregroundColor(.red.opacity(0.8))
                Spacer(minLength: 80)
            }

        case .status:
            EmptyView()
        }
    }

    // MARK: - Status

    @ViewBuilder
    private var statusIndicator: some View {
        if let stage = ws.currentStage {
            HStack(spacing: 6) {
                ProgressView()
                    .controlSize(.mini)
                    .scaleEffect(0.7)
                Text(stageLabel(stage))
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
                Spacer()
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 4)
            .transition(.opacity)
        }
    }

    private func stageLabel(_ s: String) -> String {
        switch s {
        case "sanitizing": return "Dereferencing..."
        case "glazing": return "Encrypting likeness..."
        case "thinking": return "Thinking..."
        case "restoring": return "Re-referencing..."
        default: return s
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
                    .frame(width: 36, height: 36)
                    .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                Text("Image attached")
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
                Spacer()
                Button { clearImage() } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 5)
        }
    }

    // MARK: - Input

    private var inputBar: some View {
        HStack(alignment: .center, spacing: 8) {
            Button { pickImage() } label: {
                Image(systemName: "plus.circle")
                    .font(.system(size: 18))
                    .foregroundStyle(.tertiary)
            }
            .buttonStyle(.plain)

            TextField("Message", text: $inputText)
                .textFieldStyle(.plain)
                .font(.system(size: 13))
                .padding(.horizontal, 14)
                .padding(.vertical, 9)
                .background(.white.opacity(0.05), in: Capsule())
                .onSubmit { sendMessage() }

            Button { sendMessage() } label: {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.system(size: 24))
                    .foregroundStyle(sendButtonDisabled
                        ? AnyShapeStyle(.quaternary)
                        : AnyShapeStyle(.linearGradient(
                            colors: [.blue.opacity(0.8), .purple.opacity(0.7)],
                            startPoint: .topLeading, endPoint: .bottomTrailing)))
            }
            .buttonStyle(.plain)
            .disabled(sendButtonDisabled)
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
                ws.onMessage = { handleServerMessage($0) }
            } catch {
                messages.append(ChatMessage(role: .error,
                    text: "Cannot reach server. Is python -m server.main running?",
                    model: nil, image: nil))
            }
        }
    }

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty, !isProcessing else { return }
        messages.append(ChatMessage(role: .user, text: text, model: nil, image: pendingImage))
        ws.send(text: text, model: selectedModel, imageData: pendingImageData, mimeType: pendingMimeType)
        inputText = ""
        clearImage()
        isProcessing = true
    }

    private func handleServerMessage(_ msg: AgentWebSocket.ServerMessage) {
        switch msg.type {
        case .status: break
        case .response(let text, let model, _, _):
            messages.append(ChatMessage(role: .assistant, text: text, model: model, image: nil))
            isProcessing = false
        case .error(let message):
            messages.append(ChatMessage(role: .error, text: message, model: nil, image: nil))
            isProcessing = false
        }
    }

    private func pickImage() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.png, .jpeg, .gif, .webP]
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            Task { @MainActor in
                if let data = try? Data(contentsOf: url), let image = NSImage(data: data) {
                    pendingImage = image
                    pendingImageData = data
                    let ext = url.pathExtension.lowercased()
                    pendingMimeType = ext == "jpg" || ext == "jpeg" ? "image/jpeg" :
                                      ext == "gif" ? "image/gif" :
                                      ext == "webp" ? "image/webp" : "image/png"
                }
            }
        }
    }

    private func clearImage() {
        pendingImage = nil; pendingImageData = nil; pendingMimeType = nil
    }
}
