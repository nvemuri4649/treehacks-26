//
//  AgentChatView.swift
//  Cena
//
//  Native SwiftUI chat interface — liquid glass aesthetic
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

    private let models = [
        ("Claude Sonnet 4", "claude-sonnet-4-20250514"),
        ("Claude Haiku 4", "claude-haiku-4-20250414"),
        ("GPT-4o", "gpt-4o"),
        ("GPT-4o Mini", "gpt-4o-mini"),
    ]

    var body: some View {
        VStack(spacing: 0) {
            headerBar
            glassRule
            chatArea
            statusIndicator
            imagePreviewBar
            glassRule
            inputBar
        }
        .background(.ultraThinMaterial)
        .frame(minWidth: 560, minHeight: 480)
        .onAppear { initSession() }
        .onDisappear { ws.disconnect() }
    }

    // MARK: - Thin glass separator

    private var glassRule: some View {
        Rectangle()
            .fill(.white.opacity(0.08))
            .frame(height: 1)
    }

    // MARK: - Header

    private var headerBar: some View {
        HStack(spacing: 10) {
            Image(systemName: "shield.checkered")
                .font(.system(size: 20, weight: .medium))
                .foregroundStyle(.linearGradient(
                    colors: [.blue, .purple],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ))

            Text("Cena")
                .font(.system(size: 15, weight: .semibold))

            Text("Agent")
                .font(.system(size: 15, weight: .regular))
                .foregroundStyle(.secondary)

            Spacer()

            Picker("", selection: $selectedModel) {
                ForEach(models, id: \.1) { name, id in
                    Text(name).tag(id)
                }
            }
            .pickerStyle(.menu)
            .frame(width: 170)
            .controlSize(.small)

            Button {
                initSession()
            } label: {
                Image(systemName: "arrow.counterclockwise.circle")
                    .font(.system(size: 16))
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("New session")
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
        .padding(.top, 8) // extra space for titlebar
    }

    // MARK: - Chat area

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
                    withAnimation(.easeOut(duration: 0.2)) {
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

            ZStack {
                Circle()
                    .fill(.linearGradient(
                        colors: [.blue.opacity(0.12), .purple.opacity(0.12)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ))
                    .frame(width: 80, height: 80)

                Image(systemName: "shield.checkered")
                    .font(.system(size: 36, weight: .light))
                    .foregroundStyle(.linearGradient(
                        colors: [.blue.opacity(0.7), .purple.opacity(0.7)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ))
            }

            Text("Your conversations, protected.")
                .font(.system(size: 16, weight: .semibold))

            Text("Personal data is dereferenced locally.\nLikenesses are encrypted before reaching any cloud model.")
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .lineSpacing(2)
                .frame(maxWidth: 340)

            HStack(spacing: 6) {
                pill("PII Dereferencing")
                pill("Likeness Encryption")
                pill("Local Processing")
            }
            .padding(.top, 4)

            Spacer(minLength: 60)
        }
    }

    private func pill(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 10, weight: .medium))
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(.ultraThinMaterial, in: Capsule())
            .overlay(Capsule().strokeBorder(.white.opacity(0.12), lineWidth: 0.5))
            .foregroundStyle(.secondary)
    }

    // MARK: - Bubbles

    @ViewBuilder
    private func messageBubble(_ msg: ChatMessage) -> some View {
        switch msg.role {
        case .user:
            HStack {
                Spacer(minLength: 60)
                VStack(alignment: .trailing, spacing: 4) {
                    if let img = msg.image {
                        Image(nsImage: img)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: 200, maxHeight: 140)
                            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    }
                    Text(msg.text)
                        .font(.system(size: 13))
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(
                            LinearGradient(
                                colors: [.blue.opacity(0.7), .purple.opacity(0.65)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            in: BubbleShape(isUser: true)
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
                        .background(.ultraThinMaterial, in: BubbleShape(isUser: false))
                        .overlay(
                            BubbleShape(isUser: false)
                                .stroke(.white.opacity(0.1), lineWidth: 0.5)
                        )
                        .textSelection(.enabled)

                    if let model = msg.model {
                        Text(model)
                            .font(.system(size: 10))
                            .foregroundStyle(.tertiary)
                            .padding(.leading, 6)
                    }
                }
                Spacer(minLength: 60)
            }

        case .error:
            HStack {
                Text(msg.text)
                    .font(.system(size: 12))
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(Color.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .strokeBorder(Color.red.opacity(0.2), lineWidth: 0.5)
                    )
                    .foregroundColor(.red.opacity(0.9))
                Spacer(minLength: 60)
            }

        case .status:
            EmptyView()
        }
    }

    // MARK: - Status

    @ViewBuilder
    private var statusIndicator: some View {
        if let stage = ws.currentStage {
            HStack(spacing: 8) {
                Circle()
                    .fill(stageColor(stage))
                    .frame(width: 6, height: 6)
                    .modifier(PulseAnimation())

                Text(stageLabel(stage))
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)

                Spacer()
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 5)
            .transition(.opacity.combined(with: .move(edge: .top)))
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
        case "sanitizing": return "Dereferencing personal information locally..."
        case "glazing": return "Encrypting likeness..."
        case "thinking": return "Cloud model thinking (sanitized data only)..."
        case "restoring": return "Re-referencing your information..."
        default: return s
        }
    }

    // MARK: - Image preview

    @ViewBuilder
    private var imagePreviewBar: some View {
        if let img = pendingImage {
            HStack(spacing: 10) {
                Image(nsImage: img)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 40, height: 40)
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))

                Text("Image attached")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)

                Spacer()

                Button { clearImage() } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 14))
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 6)
        }
    }

    // MARK: - Input bar

    private var inputBar: some View {
        HStack(alignment: .center, spacing: 10) {
            Button { pickImage() } label: {
                Image(systemName: "photo.on.rectangle.angled")
                    .font(.system(size: 16))
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("Attach image")

            TextField("Message...", text: $inputText)
                .textFieldStyle(.plain)
                .font(.system(size: 13))
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial, in: Capsule())
                .overlay(Capsule().strokeBorder(.white.opacity(0.1), lineWidth: 0.5))
                .onSubmit { sendMessage() }

            Button { sendMessage() } label: {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.system(size: 22))
                    .foregroundStyle(sendButtonDisabled
                        ? AnyShapeStyle(.tertiary)
                        : AnyShapeStyle(.linearGradient(
                            colors: [.blue, .purple],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                          )))
            }
            .buttonStyle(.plain)
            .disabled(sendButtonDisabled)
            .help("Send")
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
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
                ws.onMessage = { msg in handleServerMessage(msg) }
                print("🔗 Agent session: \(sid)")
            } catch {
                messages.append(ChatMessage(
                    role: .error,
                    text: "Cannot reach agent server at 127.0.0.1:8000. Is python -m server.main running?",
                    model: nil, image: nil
                ))
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
        pendingImage = nil
        pendingImageData = nil
        pendingMimeType = nil
    }
}

// MARK: - Bubble shape

struct BubbleShape: Shape {
    let isUser: Bool

    func path(in rect: CGRect) -> Path {
        let r: CGFloat = 16
        let tail: CGFloat = 4
        var path = Path()

        if isUser {
            path.addRoundedRect(in: CGRect(x: rect.minX, y: rect.minY,
                width: rect.width - tail, height: rect.height), cornerSize: CGSize(width: r, height: r))
        } else {
            path.addRoundedRect(in: CGRect(x: rect.minX + tail, y: rect.minY,
                width: rect.width - tail, height: rect.height), cornerSize: CGSize(width: r, height: r))
        }
        return path
    }
}

// MARK: - Pulse animation

struct PulseAnimation: ViewModifier {
    @State private var on = false
    func body(content: Content) -> some View {
        content.opacity(on ? 0.3 : 1.0)
            .onAppear {
                withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) { on = true }
            }
    }
}
