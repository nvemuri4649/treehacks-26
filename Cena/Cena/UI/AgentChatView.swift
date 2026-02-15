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
    @State private var expandedPrivacyId: UUID?

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
                .ignoresSafeArea()
        )
        .onAppear { initSession() }
        .onDisappear { ws.disconnect() }
    }

    private var sep: some View {
        Rectangle().fill(.white.opacity(0.07)).frame(height: 0.5)
    }

    // MARK: - Header

    private var headerBar: some View {
        HStack(spacing: 7) {
            // Logo + title grouped tight
            CenaLogo(size: 16, isAnimating: isProcessing, color: .white.opacity(0.85))

            Text("Cena")
                .font(.system(size: 14, weight: .semibold, design: .rounded))

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

            // Logo
            CenaLogo(size: 44, isAnimating: false, color: .white.opacity(0.3))
                .padding(.bottom, 6)

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

                    // Footer: model label + privacy badge
                    HStack(spacing: 8) {
                        if let model = msg.model {
                            HStack(spacing: 3) {
                                Image(systemName: models.first(where: { $0.id == model })?.icon ?? "cpu")
                                    .font(.system(size: 8))
                                Text(model)
                                    .font(.system(size: 9))
                            }
                            .foregroundStyle(.quaternary)
                        }

                        if let report = msg.privacyReport, report.totalDetected > 0 {
                            privacyBadge(msg: msg, report: report)
                        }
                    }
                    .padding(.leading, 6)

                    // Expandable privacy detail panel
                    if expandedPrivacyId == msg.id, let report = msg.privacyReport {
                        privacyDetailPanel(report: report, sanitizedPrompt: msg.sanitizedPrompt)
                            .transition(.opacity.combined(with: .move(edge: .top)))
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
            VStack(spacing: 4) {
                PipelineView(stage: stage)
                PipelineStageLabel(stage: stage)
            }
            .padding(.vertical, 4)
            .transition(.opacity.combined(with: .scale(scale: 0.95)))
            .animation(.easeInOut(duration: 0.3), value: stage)
        }
    }

    // MARK: - Privacy Badge

    private func privacyBadge(msg: ChatMessage, report: PrivacyReport) -> some View {
        Button {
            withAnimation(.easeInOut(duration: 0.2)) {
                expandedPrivacyId = (expandedPrivacyId == msg.id) ? nil : msg.id
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: "shield.checkered")
                    .font(.system(size: 8, weight: .semibold))
                Text("\(report.protectedCount) protected")
                    .font(.system(size: 9, weight: .medium))
                Image(systemName: expandedPrivacyId == msg.id ? "chevron.up" : "chevron.down")
                    .font(.system(size: 6, weight: .bold))
            }
            .foregroundStyle(privacyConfidenceColor(report.privacyConfidence))
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(
                privacyConfidenceColor(report.privacyConfidence).opacity(0.12),
                in: Capsule()
            )
            .overlay(
                Capsule().stroke(
                    privacyConfidenceColor(report.privacyConfidence).opacity(0.2),
                    lineWidth: 0.5
                )
            )
        }
        .buttonStyle(.plain)
        .help("Privacy: \(report.protectedCount) items protected, \(Int(report.privacyConfidence * 100))% confidence")
    }

    private func privacyConfidenceColor(_ confidence: Double) -> Color {
        if confidence >= 0.85 { return .green }
        if confidence >= 0.60 { return .yellow }
        return .orange
    }

    // MARK: - Privacy Detail Panel

    private func privacyDetailPanel(report: PrivacyReport, sanitizedPrompt: String?) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            // Confidence bar
            HStack(spacing: 8) {
                Image(systemName: "shield.lefthalf.filled")
                    .font(.system(size: 11))
                    .foregroundStyle(privacyConfidenceColor(report.privacyConfidence))

                VStack(alignment: .leading, spacing: 3) {
                    Text("Privacy Confidence: \(Int(report.privacyConfidence * 100))%")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.primary.opacity(0.8))

                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 2)
                                .fill(.white.opacity(0.08))
                                .frame(height: 4)
                            RoundedRectangle(cornerRadius: 2)
                                .fill(privacyConfidenceColor(report.privacyConfidence))
                                .frame(width: geo.size.width * report.privacyConfidence, height: 4)
                        }
                    }
                    .frame(height: 4)
                }
            }

            // Stats row
            HStack(spacing: 12) {
                statPill(count: report.redactedCount, label: "Redacted", color: .red)
                statPill(count: report.blurredCount, label: "Blurred", color: .yellow)
                statPill(count: report.keptCount, label: "Kept", color: .green)
            }

            // Entity list
            if !report.entities.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Detected Entities")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .textCase(.uppercase)

                    ForEach(report.entities) { entity in
                        HStack(spacing: 6) {
                            Image(systemName: entity.actionIcon)
                                .font(.system(size: 8))
                                .foregroundColor(entityColor(entity.action))
                                .frame(width: 12)

                            Text(entity.typeLabel)
                                .font(.system(size: 10, weight: .medium))
                                .foregroundStyle(.primary.opacity(0.7))

                            Spacer()

                            if let hint = entity.hint {
                                Text("~ \(hint)")
                                    .font(.system(size: 9))
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                            }

                            Text(entity.action)
                                .font(.system(size: 8, weight: .semibold, design: .monospaced))
                                .foregroundColor(entityColor(entity.action))
                                .padding(.horizontal, 5)
                                .padding(.vertical, 1)
                                .background(entityColor(entity.action).opacity(0.12), in: Capsule())
                        }
                    }
                }
            }

            // Warnings
            if !report.warnings.isEmpty {
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(report.warnings, id: \.self) { warning in
                        HStack(alignment: .top, spacing: 4) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .font(.system(size: 8))
                                .foregroundColor(.orange)
                            Text(warning)
                                .font(.system(size: 9))
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            }

            // Sanitized prompt preview
            if let prompt = sanitizedPrompt, !prompt.isEmpty {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Sent to Cloud")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .textCase(.uppercase)

                    Text(prompt)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.primary.opacity(0.5))
                        .lineLimit(4)
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.black.opacity(0.2), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                        .textSelection(.enabled)
                }
            }
        }
        .padding(12)
        .background(.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(.white.opacity(0.06), lineWidth: 0.5)
        )
        .padding(.top, 4)
    }

    private func statPill(count: Int, label: String, color: Color) -> some View {
        HStack(spacing: 3) {
            Text("\(count)")
                .font(.system(size: 10, weight: .bold, design: .rounded))
                .foregroundColor(color)
            Text(label)
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(color.opacity(0.08), in: Capsule())
    }

    private func entityColor(_ action: String) -> Color {
        switch action {
        case "REDACT": return .red
        case "BLUR":   return .yellow
        case "KEEP":   return .green
        default:       return .gray
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
                messages.append(ChatMessage(
                    role: .error,
                    text: "Cannot reach server. Is python -m server.main running?"
                ))
            }
        }
    }

    private func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespaces)
        guard !text.isEmpty, !isProcessing else { return }
        messages.append(ChatMessage(role: .user, text: text, image: pendingImage))
        ws.send(text: text, model: selectedModel, imageData: pendingImageData, mimeType: pendingMimeType)
        inputText = ""; clearImage(); isProcessing = true
    }

    private func handleServerMessage(_ msg: AgentWebSocket.ServerMessage) {
        switch msg.type {
        case .status: break
        case .response(let text, let model, _, let sanitizedPrompt, let privacyReport):
            messages.append(ChatMessage(
                role: .assistant,
                text: text,
                model: model,
                sanitizedPrompt: sanitizedPrompt,
                privacyReport: privacyReport
            ))
            isProcessing = false
        case .error(let message):
            messages.append(ChatMessage(role: .error, text: message))
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
