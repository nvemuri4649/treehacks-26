//
//  AgentChatView.swift
//  Cena
//
//  Translucent glass chat interface with accent colors
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

    private let accentGrad = LinearGradient(
        colors: [.blue, .purple],
        startPoint: .topLeading, endPoint: .bottomTrailing
    )

    private let models: [(icon: String, name: String, id: String)] = [
        ("sparkle", "Claude Sonnet 4", "claude-sonnet-4-20250514"),
        ("hare", "Claude Haiku 4", "claude-haiku-4-20250414"),
        ("brain", "GPT-4o", "gpt-4o"),
        ("bolt", "GPT-4o Mini", "gpt-4o-mini"),
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
        .frame(minWidth: 640, minHeight: 480)
        .background(
            VisualEffectViewRepresentable(material: .hudWindow, blendingMode: .behindWindow)
        )
        .onAppear { initSession() }
        .onDisappear { ws.disconnect() }
    }

    private var sep: some View {
        Rectangle().fill(.white.opacity(0.07)).frame(height: 0.5)
    }

    // MARK: - Header

    private var headerBar: some View {
        HStack(spacing: 8) {
            // Icon
            Image(systemName: "shield.checkered")
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(accentGrad)

            Text("Cena")
                .font(.system(size: 14, weight: .semibold, design: .rounded))

            Text("·")
                .foregroundStyle(.quaternary)

            Text("Agent")
                .font(.system(size: 13, weight: .regular))
                .foregroundStyle(.secondary)

            Spacer()

            modelSelector

            // New session
            Button { initSession() } label: {
                Image(systemName: "plus.message")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(.secondary)
                    .padding(6)
                    .background(.white.opacity(0.07), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                    .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(.white.opacity(0.08), lineWidth: 0.5))
            }
            .buttonStyle(.plain)
            .help("New session")
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 11)
        .padding(.top, 14)
    }

    // MARK: - Model selector

    private var selectedModelIcon: String {
        models.first(where: { $0.id == selectedModel })?.icon ?? "cpu"
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
            HStack(spacing: 6) {
                Image(systemName: selectedModelIcon)
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(accentGrad)
                Text(selectedModelName)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.primary.opacity(0.7))
                Image(systemName: "chevron.down")
                    .font(.system(size: 7, weight: .bold))
                    .foregroundStyle(.quaternary)
            }
            .padding(.horizontal, 11)
            .padding(.vertical, 6)
            .background(.white.opacity(0.07), in: Capsule())
            .overlay(Capsule().stroke(.white.opacity(0.08), lineWidth: 0.5))
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
    }

    // MARK: - Chat

    private var chatArea: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 14) {
                    if messages.isEmpty { welcomeView }
                    ForEach(messages) { msg in
                        messageBubble(msg).id(msg.id)
                    }
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 14)
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

    // MARK: - Welcome

    private var welcomeView: some View {
        VStack(spacing: 14) {
            Spacer(minLength: 60)

            // Subtle glow icon
            ZStack {
                Circle()
                    .fill(accentGrad.opacity(0.1))
                    .frame(width: 64, height: 64)
                Image(systemName: "shield.checkered")
                    .font(.system(size: 28, weight: .light))
                    .foregroundStyle(accentGrad.opacity(0.6))
            }

            Text("What can I help with?")
                .font(.system(size: 20, weight: .medium, design: .rounded))
                .foregroundStyle(.primary.opacity(0.6))

            Text("Personal data is processed locally before reaching the cloud.")
                .font(.system(size: 11))
                .foregroundStyle(.secondary.opacity(0.7))
                .multilineTextAlignment(.center)
                .frame(maxWidth: 320)

            Spacer(minLength: 60)
        }
    }

    // MARK: - Bubbles

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
                            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                    }
                    Text(msg.text)
                        .font(.system(size: 13))
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(
                            accentGrad.opacity(0.6),
                            in: RoundedRectangle(cornerRadius: 18, style: .continuous)
                        )
                        .foregroundColor(.white)
                }
            }

        case .assistant:
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text(msg.text)
                        .font(.system(size: 13))
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(.white.opacity(0.08),
                                    in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 18, style: .continuous)
                                .stroke(.white.opacity(0.06), lineWidth: 0.5)
                        )
                        .textSelection(.enabled)

                    if let model = msg.model {
                        HStack(spacing: 3) {
                            Image(systemName: models.first(where: { $0.id == model })?.icon ?? "cpu")
                                .font(.system(size: 8))
                            Text(model)
                                .font(.system(size: 9))
                        }
                        .foregroundStyle(.quaternary)
                        .padding(.leading, 6)
                    }
                }
                Spacer(minLength: 80)
            }

        case .error:
            HStack {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.system(size: 10))
                    Text(msg.text)
                        .font(.system(size: 12))
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(Color.red.opacity(0.1),
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
            HStack(spacing: 7) {
                Circle()
                    .fill(stageColor(stage))
                    .frame(width: 6, height: 6)
                    .opacity(isProcessing ? 1 : 0.5)
                    .animation(.easeInOut(duration: 0.6).repeatForever(autoreverses: true), value: isProcessing)
                Text(stageLabel(stage))
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
                Spacer()
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 5)
            .transition(.opacity)
        }
    }

    private func stageColor(_ s: String) -> Color {
        switch s {
        case "sanitizing": return .yellow
        case "glazing": return .purple
        case "thinking": return .blue
        case "restoring": return .green
        default: return .gray
        }
    }

    private func stageLabel(_ s: String) -> String {
        switch s {
        case "sanitizing": return "Dereferencing personal information..."
        case "glazing": return "Encrypting likeness..."
        case "thinking": return "Thinking..."
        case "restoring": return "Re-referencing your information..."
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
                    .foregroundStyle(.secondary)
                Spacer()
                Button { clearImage() } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 13))
                        .foregroundStyle(.quaternary)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 5)
        }
    }

    // MARK: - Input

    private var inputBar: some View {
        HStack(alignment: .center, spacing: 10) {
            Button { pickImage() } label: {
                Image(systemName: "photo.on.rectangle.angled")
                    .font(.system(size: 15))
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("Attach image")

            TextField("Message...", text: $inputText)
                .textFieldStyle(.plain)
                .font(.system(size: 13))
                .padding(.horizontal, 14)
                .padding(.vertical, 9)
                .background(.white.opacity(0.07), in: Capsule())
                .overlay(Capsule().stroke(.white.opacity(0.06), lineWidth: 0.5))
                .onSubmit { sendMessage() }

            Button { sendMessage() } label: {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.system(size: 24))
                    .foregroundStyle(sendButtonDisabled
                        ? AnyShapeStyle(.quaternary)
                        : AnyShapeStyle(accentGrad))
            }
            .buttonStyle(.plain)
            .disabled(sendButtonDisabled)
            .help("Send")
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
    }

    private var sendButtonDisabled: Bool {
        isProcessing || inputText.trimmingCharacters(in: .whitespaces).isEmpty
    }

    // MARK: - Actions

    private func initSession() {
        messages = []; ws.disconnect(); isProcessing = false; clearImage()
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
        inputText = ""; clearImage(); isProcessing = true
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
                    pendingImage = image; pendingImageData = data
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
