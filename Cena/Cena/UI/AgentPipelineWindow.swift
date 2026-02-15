//
// AgentPipelineWindow.swift
// Cena
//
// A polished, animated visualization of the local ↔ cloud agent pipeline.
// Opens as a companion window to the left of the chat.
//
// Flow (vertical):
//   You  →  Local Guardian (Nemotron)  →  Cloud LLM  →  Local Re-ref  →  You
//

import SwiftUI

//MARK: - Observable stage model (shared with chat)

class PipelineStageModel: ObservableObject {
    @Published var currentStage: String? = nil   // nil = idle
    @Published var isActive: Bool = false
}

//MARK: - Main view

struct AgentPipelineWindowView: View {
    @ObservedObject var model: PipelineStageModel

    //Animation state
    @State private var particlePhase: CGFloat = 0
    @State private var pulseScale: CGFloat = 1.0
    @State private var glowOpacity: Double = 0.3

    private let nodes: [(id: String, icon: String, label: String, sub: String, stage: String?)] = [
        ("user",    "person.fill",       "You",             "Raw message + PII",          nil),
        ("local1",  "cpu",               "Local Guardian",  "Nemotron · PII Analysis",    "sanitizing"),
        ("encrypt", "shield.checkered",  "Likeness Shield", "Encryption",                 "encrypting"),
        ("cloud",   "cloud.fill",        "Cloud LLM",       "Claude · GPT · Reasoning",   "thinking"),
        ("local2",  "arrow.uturn.left",  "Re-Reference",    "Restore PII Tokens",         "restoring"),
        ("result",  "checkmark.seal.fill","Secure Response", "Privacy-safe answer",        nil),
    ]

    var body: some View {
        VStack(spacing: 0) {
            //Header
            header
            Spacer(minLength: 12)
            //Pipeline nodes
            pipelineStack
            Spacer(minLength: 12)
            //Footer
            footer
        }
        .frame(width: 260)
        .padding(.vertical, 16)
        .background(
            VisualEffectViewRepresentable(material: .hudWindow, blendingMode: .behindWindow)
                .ignoresSafeArea()
        )
        .onAppear { startAnimations() }
    }

    //MARK: - Header

    private var header: some View {
        VStack(spacing: 6) {
            CenaLogo(size: 22, isAnimating: model.isActive, color: .white.opacity(0.7))
            Text("Privacy Pipeline")
                .font(.system(size: 13, weight: .semibold, design: .rounded))
                .foregroundStyle(.primary.opacity(0.8))
            Text(model.isActive ? "Processing..." : "Idle")
                .font(.system(size: 10))
                .foregroundStyle(.secondary.opacity(0.6))
        }
        .padding(.horizontal, 20)
        .padding(.top, 16)
    }

    //MARK: - Pipeline stack

    private var pipelineStack: some View {
        VStack(spacing: 0) {
            ForEach(Array(nodes.enumerated()), id: \.element.id) { index, node in
                //Node
                nodeRow(node: node, index: index)

                //Connector (except after last)
                if index < nodes.count - 1 {
                    connectorView(
                        fromStage: node.stage,
                        toStage: nodes[index + 1].stage,
                        index: index
                    )
                }
            }
        }
        .padding(.horizontal, 20)
    }

    //MARK: - Node row

    private func nodeRow(node: (id: String, icon: String, label: String, sub: String, stage: String?), index: Int) -> some View {
        let isActive = node.stage != nil && model.currentStage == node.stage
        let isPast = isPastStage(node.stage)
        let isFutureOrIdle = !isActive && !isPast

        return HStack(spacing: 12) {
            //Glowing orb
            ZStack {
                //Glow ring (active only)
                if isActive {
                    Circle()
                        .fill(stageColor(node.stage).opacity(0.15))
                        .frame(width: 44, height: 44)
                        .scaleEffect(pulseScale)

                    Circle()
                        .stroke(stageColor(node.stage).opacity(glowOpacity), lineWidth: 2)
                        .frame(width: 44, height: 44)
                        .scaleEffect(pulseScale)
                }

                //Past checkmark ring
                if isPast {
                    Circle()
                        .stroke(Color.green.opacity(0.3), lineWidth: 1.5)
                        .frame(width: 38, height: 38)
                }

                //Core circle
                Circle()
                    .fill(
                        isActive ? stageColor(node.stage).opacity(0.25) :
                        isPast ? Color.green.opacity(0.08) :
                        Color.white.opacity(0.05)
                    )
                    .frame(width: 36, height: 36)

                //Icon
                if isPast && node.stage != nil {
                    Image(systemName: "checkmark")
                        .font(.system(size: 13, weight: .bold))
                        .foregroundStyle(Color.green.opacity(0.7))
                } else {
                    Image(systemName: node.icon)
                        .font(.system(size: 14, weight: .medium))
                        .foregroundStyle(
                            isActive ? stageColor(node.stage) :
                            isFutureOrIdle ? .secondary.opacity(0.4) :
                            .secondary.opacity(0.7)
                        )
                        .symbolEffect(.pulse, options: .repeating, isActive: isActive)
                }
            }
            .frame(width: 44, height: 44)

            //Labels
            VStack(alignment: .leading, spacing: 2) {
                Text(node.label)
                    .font(.system(size: 12, weight: isActive ? .bold : .medium))
                    .foregroundColor(
                        isActive ? .white :
                        isPast ? .white.opacity(0.6) :
                        .white.opacity(0.35)
                    )

                Text(node.sub)
                    .font(.system(size: 9))
                    .foregroundColor(
                        isActive ? stageColor(node.stage).opacity(0.8) :
                        .white.opacity(0.3)
                    )
                    .lineLimit(1)
            }

            Spacer()
        }
    }

    //MARK: - Connector (animated particles between nodes)

    private func connectorView(fromStage: String?, toStage: String?, index: Int) -> some View {
        let isFlowing = isConnectorActive(fromStage: fromStage, toStage: toStage)
        let isPast = isConnectorPast(fromStage: fromStage, toStage: toStage)
        let color = connectorColor(toStage: toStage)

        return HStack(spacing: 0) {
            Spacer().frame(width: 20) // Align under node center

            ZStack {
                //Base line
                RoundedRectangle(cornerRadius: 1)
                    .fill(isPast ? Color.green.opacity(0.15) : Color.white.opacity(0.04))
                    .frame(width: 2, height: 24)

                //Active glow line
                if isFlowing {
                    RoundedRectangle(cornerRadius: 1)
                        .fill(color.opacity(0.4))
                        .frame(width: 2, height: 24)
                }

                //Flowing particles
                if isFlowing {
                    ForEach(0..<3, id: \.self) { i in
                        let phase = (particlePhase + CGFloat(i) * 0.33)
                            .truncatingRemainder(dividingBy: 1.0)
                        Circle()
                            .fill(color.opacity(0.9))
                            .frame(width: 5, height: 5)
                            .shadow(color: color.opacity(0.5), radius: 4)
                            .offset(y: -12 + phase * 24)
                    }
                }
            }
            .frame(width: 44, height: 24)

            Spacer()
        }
    }

    //MARK: - Footer

    private var footer: some View {
        VStack(spacing: 4) {
            if let stage = model.currentStage {
                HStack(spacing: 6) {
                    Circle()
                        .fill(stageColor(stage))
                        .frame(width: 6, height: 6)
                        .shadow(color: stageColor(stage).opacity(0.5), radius: 3)

                    Text(stageDescription(stage))
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(.secondary)
                }
                .transition(.opacity.combined(with: .scale(scale: 0.9)))
                .animation(.easeInOut(duration: 0.25), value: stage)
            } else {
                Text("Personal data never leaves unprotected")
                    .font(.system(size: 9))
                    .foregroundStyle(.quaternary)
            }
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 8)
    }

    //MARK: - Helpers

    private func stageColor(_ stage: String?) -> Color {
        switch stage {
        case "sanitizing": return .yellow
        case "encrypting":    return .purple
        case "thinking":   return .blue
        case "restoring":  return .green
        default:           return .white
        }
    }

    private func stageDescription(_ stage: String) -> String {
        switch stage {
        case "sanitizing": return "Analyzing & redacting PII locally..."
        case "encrypting":    return "Encrypting likeness locally..."
        case "thinking":   return "Cloud LLM reasoning on safe data..."
        case "restoring":  return "Restoring PII tokens locally..."
        default:           return stage
        }
    }

    private var stageOrder: [String] { ["sanitizing", "encrypting", "thinking", "restoring"] }

    private func stageIdx(_ s: String?) -> Int {
        guard let s = s else { return -1 }
        return stageOrder.firstIndex(of: s) ?? -1
    }

    private func isPastStage(_ stage: String?) -> Bool {
        guard model.isActive, let current = model.currentStage else { return false }
        return stageIdx(stage) < stageIdx(current) && stageIdx(stage) >= 0
    }

    private func isConnectorActive(fromStage: String?, toStage: String?) -> Bool {
        guard model.isActive, let current = model.currentStage else { return false }
        let ci = stageIdx(current)
        let fi = stageIdx(fromStage)
        //Connector is active when we're at or transitioning to the toStage
        return fi >= 0 && fi < ci || (fromStage == nil && ci >= 0) || stageIdx(toStage) == ci
    }

    private func isConnectorPast(fromStage: String?, toStage: String?) -> Bool {
        guard model.isActive, let current = model.currentStage else { return false }
        return stageIdx(toStage) < stageIdx(current) && stageIdx(toStage) >= 0
    }

    private func connectorColor(toStage: String?) -> Color {
        stageColor(toStage ?? model.currentStage)
    }

    //MARK: - Animations

    private func startAnimations() {
        withAnimation(.linear(duration: 1.0).repeatForever(autoreverses: false)) {
            particlePhase = 1.0
        }
        withAnimation(.easeInOut(duration: 1.5).repeatForever(autoreverses: true)) {
            pulseScale = 1.12
        }
        withAnimation(.easeInOut(duration: 1.2).repeatForever(autoreverses: true)) {
            glowOpacity = 0.7
        }
    }
}
