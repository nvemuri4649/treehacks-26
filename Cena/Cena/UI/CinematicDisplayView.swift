//
//  CinematicDisplayView.swift
//  Cena
//
//  Side-by-side display of original and encrypted demo images.
//

import SwiftUI

struct CinematicDisplayView: View {
    private let dir: URL? = {
        let p = URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Documents/StudioProjects/treehacks-26/output")
        return FileManager.default.fileExists(atPath: p.appendingPathComponent("demo_original.png").path) ? p : nil
    }()

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                CenaLogo(size: 16, isAnimating: false, color: .white.opacity(0.7))
                Text("Cena")
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                Spacer()
            }
            .padding(.horizontal, 20)
            .padding(.top, 24)
            .padding(.bottom, 10)

            Rectangle().fill(.white.opacity(0.06)).frame(height: 0.5)

            VStack(spacing: 14) {
                imageBlock(filename: "demo_original.png", label: "Original", badgeColor: .red)
                imageBlock(filename: "demo_glazed.png", label: "Encrypted", badgeColor: .green)
            }
            .padding(16)

            Spacer(minLength: 0)
        }
        .frame(width: 480)
        .background(
            VisualEffectViewRepresentable(material: .hudWindow, blendingMode: .behindWindow)
                .ignoresSafeArea()
        )
    }

    private func imageBlock(filename: String, label: String, badgeColor: Color) -> some View {
        ZStack(alignment: .topLeading) {
            if let url = dir?.appendingPathComponent(filename),
               let img = NSImage(contentsOf: url) {
                Image(nsImage: img)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .stroke(.white.opacity(0.08), lineWidth: 0.5)
                    )
            } else {
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(.white.opacity(0.04))
                    .aspectRatio(1, contentMode: .fit)
            }

            HStack(spacing: 4) {
                Circle().fill(badgeColor).frame(width: 6, height: 6)
                Text(label)
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundColor(.white.opacity(0.9))
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(.ultraThinMaterial, in: Capsule())
            .overlay(Capsule().stroke(.white.opacity(0.1), lineWidth: 0.5))
            .padding(8)
        }
    }
}
