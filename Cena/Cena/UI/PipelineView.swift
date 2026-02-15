//
//  PipelineView.swift
//  Cena
//
//  Animated visualization of the local-cloud agent pipeline
//

import SwiftUI

struct PipelineView: View {
    let stage: String  // "sanitizing", "glazing", "thinking", "restoring"

    @State private var particleOffset: CGFloat = 0

    private var stageIndex: Int {
        switch stage {
        case "sanitizing": return 0
        case "glazing": return 1
        case "thinking": return 2
        case "restoring": return 3
        default: return 0
        }
    }

    var body: some View {
        HStack(spacing: 0) {
            // Local agent node
            nodeView(
                icon: "desktopcomputer",
                label: "Local",
                isActive: stageIndex <= 1
            )

            // Arrow with particles flowing
            animatedArrow(
                flowRight: stageIndex == 2,
                flowLeft: stageIndex == 3,
                color: stageIndex == 2 ? .blue : stageIndex == 3 ? .green : .clear
            )

            // Cloud node
            nodeView(
                icon: "cloud",
                label: "Cloud",
                isActive: stageIndex == 2
            )

            // Arrow back
            animatedArrow(
                flowRight: stageIndex == 3,
                flowLeft: false,
                color: stageIndex == 3 ? .green : .clear
            )
            .opacity(0) // hidden, keeps layout balanced

            Spacer(minLength: 0)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 8)
        .onAppear {
            withAnimation(.linear(duration: 1.2).repeatForever(autoreverses: false)) {
                particleOffset = 1.0
            }
        }
    }

    // MARK: - Node

    private func nodeView(icon: String, label: String, isActive: Bool) -> some View {
        VStack(spacing: 3) {
            ZStack {
                Circle()
                    .fill(isActive ? .white.opacity(0.1) : .white.opacity(0.04))
                    .frame(width: 32, height: 32)

                Image(systemName: icon)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(isActive ? .primary : .tertiary)
            }
            Text(label)
                .font(.system(size: 9, weight: .medium))
                .foregroundStyle(isActive ? .secondary : .quaternary)
        }
    }

    // MARK: - Animated arrow with flowing dots

    private func animatedArrow(flowRight: Bool, flowLeft: Bool, color: Color) -> some View {
        GeometryReader { geo in
            let w = geo.size.width
            ZStack {
                // Base line
                Rectangle()
                    .fill(.white.opacity(0.06))
                    .frame(height: 1)
                    .offset(y: 0)

                // Flowing particles
                if flowRight || flowLeft {
                    ForEach(0..<3, id: \.self) { i in
                        let basePhase = (particleOffset + CGFloat(i) * 0.33).truncatingRemainder(dividingBy: 1.0)
                        let xPos = flowRight ? basePhase * w : (1 - basePhase) * w
                        Circle()
                            .fill(color.opacity(0.8))
                            .frame(width: 4, height: 4)
                            .offset(x: xPos - w / 2)
                    }
                }
            }
            .frame(height: 32) // match node height
        }
        .frame(width: 60, height: 32)
    }
}

// MARK: - Stage descriptor

struct PipelineStageLabel: View {
    let stage: String

    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(stageColor)
                .frame(width: 5, height: 5)
            Text(stageText)
                .font(.system(size: 10))
                .foregroundStyle(.secondary)
        }
    }

    private var stageColor: Color {
        switch stage {
        case "sanitizing": return .yellow
        case "glazing": return .purple
        case "thinking": return .blue
        case "restoring": return .green
        default: return .gray
        }
    }

    private var stageText: String {
        switch stage {
        case "sanitizing": return "Dereferencing locally..."
        case "glazing": return "Encrypting likeness..."
        case "thinking": return "Cloud reasoning..."
        case "restoring": return "Re-referencing..."
        default: return stage
        }
    }
}
