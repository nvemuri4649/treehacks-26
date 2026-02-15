//
// EncryptionProgressView.swift
// Cena
//
// encryption progress — image with layered visual effects
//

import SwiftUI

struct EncryptionProgressView: View {
    @ObservedObject var job: EncryptionJob
    let onCancel: () -> Void

    //Animation drivers
    @State private var glitchX: CGFloat = 0
    @State private var hueShift: Double = 0
    @State private var vignettePulse: Double = 0.25
    @State private var edgeGlow: Double = 0.0

    private let accent = LinearGradient(
        colors: [.blue, .purple],
        startPoint: .topLeading, endPoint: .bottomTrailing
    )

    var body: some View {
        ZStack {
            VisualEffectViewRepresentable(material: .hudWindow, blendingMode: .behindWindow)
                .cornerRadius(24)
                .shadow(color: .black.opacity(0.3), radius: 20, x: 0, y: 10)

            VStack(spacing: 0) {
                imageSection
                    .frame(height: 340)
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .clipped()

                controlsSection
            }
            .padding(20)
        }
        .frame(width: 480, height: 580)
        .onAppear { startAnimations() }
    }

    //MARK: - Image with layered FX

    private var imageSection: some View {
        ZStack {
            //Base image
            if let cg = job.originalImage.cgImage(forProposedRect: nil, context: nil, hints: nil) {
                Image(decorative: cg, scale: 1.0)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(height: 340)
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .stroke(.white.opacity(0.08), lineWidth: 0.5)
                    )
                    .offset(x: isProcessing ? glitchX : 0)
                    .hueRotation(.degrees(isProcessing ? hueShift : 0))
            } else {
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(.white.opacity(0.04))
                    .overlay(CenaLogo(size: 48, isAnimating: true, color: .white.opacity(0.25)))
            }

            if isProcessing {
                //── Face outline (if mask available) ──
                if isFaceMode, let mask = job.maskImage,
                   let maskCG = mask.cgImage(forProposedRect: nil, context: nil, hints: nil) {
                    GeometryReader { geo in
                        let faceRect = extractFaceRect(from: maskCG, in: geo.size)
                        if let fr = faceRect {
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .stroke(
                                    LinearGradient(colors: [.cyan.opacity(0.5), .blue.opacity(0.4)],
                                                   startPoint: .top, endPoint: .bottom),
                                    lineWidth: 1.5
                                )
                                .frame(width: fr.width, height: fr.height)
                                .position(x: fr.midX, y: fr.midY)
                                .shadow(color: .cyan.opacity(0.3), radius: 8)
                        }
                    }
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                }

                //── Fine grain sand noise (concentrated on face if mask) ──
                TimelineView(.animation(minimumInterval: 0.15)) { timeline in
                    Canvas { ctx, size in
                        let t = timeline.date.timeIntervalSinceReferenceDate
                        let faceR = faceNormRect
                        let count = 3200
                        for i in 0..<count {
                            let h1 = sin(Double(i) * 127.1 + t * 1.9) * 43758.5453
                            let h2 = sin(Double(i) * 269.5 + t * 1.3) * 28461.3217
                            let h3 = sin(Double(i) * 78.3 + t * 2.7) * 17539.7621

                            var px = (h1 - floor(h1)) * Double(size.width)
                            var py = (h2 - floor(h2)) * Double(size.height)
                            let raw = h3 - floor(h3)

                            //70% of particles cluster inside face region
                            if let fr = faceR, raw < 0.7 {
                                px = (fr.minX + (h1 - floor(h1)) * fr.width) * Double(size.width)
                                py = (fr.minY + (h2 - floor(h2)) * fr.height) * Double(size.height)
                            }

                            let r = 0.4 + raw * 0.8
                            let inFace = faceR.map { px >= $0.minX * Double(size.width) && px <= $0.maxX * Double(size.width) && py >= $0.minY * Double(size.height) && py <= $0.maxY * Double(size.height) } ?? false
                            let alpha = inFace ? (0.08 + raw * 0.18) : (0.03 + raw * 0.07)
                            let rect = CGRect(x: px - r / 2, y: py - r / 2, width: r, height: r)
                            ctx.fill(Path(ellipseIn: rect), with: .color(.white.opacity(alpha)))
                        }
                    }
                }
                .blendMode(.screen)
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))

                //── Edge glow ──
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .stroke(
                        LinearGradient(colors: [.blue.opacity(edgeGlow), .purple.opacity(edgeGlow), .cyan.opacity(edgeGlow * 0.5)],
                                       startPoint: .topLeading, endPoint: .bottomTrailing),
                        lineWidth: 2
                    )
                    .blur(radius: 4)

                //── Layer 6: Vignette ──
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(
                        RadialGradient(
                            colors: [.clear, .black.opacity(vignettePulse)],
                            center: .center,
                            startRadius: 80,
                            endRadius: 260
                        )
                    )
            }

            //Status badge
            VStack {
                HStack {
                    Spacer()
                    HStack(spacing: 6) {
                        CenaLogo(size: 14, isAnimating: isProcessing, color: .white.opacity(0.9))
                        Text(statusText)
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundColor(.white)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(.ultraThinMaterial, in: Capsule())
                    .overlay(Capsule().stroke(.white.opacity(0.15), lineWidth: 0.5))
                }
                .padding(12)
                Spacer()
            }

            //Completed overlay
            if isComplete {
                ZStack {
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(.green.opacity(0.12))

                    VStack(spacing: 10) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 52))
                            .foregroundStyle(.green)
                            .shadow(color: .green.opacity(0.4), radius: 12)
                        Text("Encrypted")
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundColor(.white)
                    }
                }
                .transition(.opacity.combined(with: .scale(scale: 0.95)))
            }
        }
    }

    //MARK: - Controls

    private var controlsSection: some View {
        VStack(spacing: 14) {
            Spacer().frame(height: 6)

            VStack(spacing: 8) {
                HStack {
                    Text("\(Int(job.progress * 100))%")
                        .font(.system(size: 13, weight: .bold, design: .rounded))
                        .foregroundColor(.white)
                    Spacer()
                    Text("\(job.currentIteration) / \(job.totalIterations)")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.white.opacity(0.5))
                }

                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 6)
                            .fill(Color.white.opacity(0.08))
                            .frame(height: 8)

                        RoundedRectangle(cornerRadius: 6)
                            .fill(accent)
                            .frame(width: geo.size.width * CGFloat(job.progress), height: 8)
                            .shadow(color: .blue.opacity(0.4), radius: 6, y: 2)
                            .animation(.easeInOut(duration: 0.3), value: job.progress)
                    }
                }
                .frame(height: 8)

                HStack {
                    HStack(spacing: 4) {
                        Image(systemName: "clock")
                            .font(.system(size: 9))
                        Text(job.estimatedTimeRemaining())
                            .font(.system(size: 10))
                    }
                    .foregroundColor(.white.opacity(0.4))

                    Spacer()

                    if let msg = statusMessage {
                        Text(msg)
                            .font(.system(size: 10))
                            .foregroundColor(.white.opacity(0.4))
                    }
                }
            }

            Button(action: onCancel) {
                Text("Cancel")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundColor(.white.opacity(0.6))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                    .background(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(.white.opacity(0.05))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .stroke(.white.opacity(0.07), lineWidth: 0.5)
                    )
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 4)
    }

    //MARK: - State

    private var isProcessing: Bool {
        switch job.status {
        case .protecting, .generatingMask: return true
        default: return false
        }
    }

    private var isComplete: Bool {
        if case .completed = job.status { return true }
        return false
    }

    private var statusText: String {
        switch job.status {
        case .pending: return "Preparing…"
        case .generatingMask: return "Detecting Faces"
        case .protecting: return "Encrypting"
        case .completed: return "Complete"
        case .failed: return "Failed"
        case .cancelled: return "Cancelled"
        }
    }

    private var statusMessage: String? {
        switch job.status {
        case .generatingMask: return "Vision face detection"
        case .protecting where job.currentIteration > 0: return "Adversarial perturbations"
        case .completed: return "Ready to use"
        case .failed(let e): return e.localizedDescription
        default: return nil
        }
    }

    //MARK: - Face helpers

    private var isFaceMode: Bool {
        if case .autoFace = job.maskMode { return true }
        return false
    }

    private var faceNormRect: CGRect? {
        guard isFaceMode, let mask = job.maskImage,
              let cg = mask.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }
        return extractNormFaceRect(from: cg)
    }

    private func extractNormFaceRect(from cg: CGImage) -> CGRect? {
        guard let data = cg.dataProvider?.data, let ptr = CFDataGetBytePtr(data) else { return nil }
        let w = cg.width, h = cg.height, bpr = cg.bytesPerRow, bpp = cg.bitsPerPixel / 8
        var minX = w, minY = h, maxX = 0, maxY = 0
        for y in stride(from: 0, to: h, by: 4) {
            for x in stride(from: 0, to: w, by: 4) {
                if ptr[y * bpr + x * bpp] > 128 {
                    minX = min(minX, x); minY = min(minY, y)
                    maxX = max(maxX, x); maxY = max(maxY, y)
                }
            }
        }
        guard maxX > minX, maxY > minY else { return nil }
        return CGRect(
            x: CGFloat(minX) / CGFloat(w),
            y: CGFloat(minY) / CGFloat(h),
            width: CGFloat(maxX - minX) / CGFloat(w),
            height: CGFloat(maxY - minY) / CGFloat(h)
        )
    }

    private func extractFaceRect(from cg: CGImage, in viewSize: CGSize) -> CGRect? {
        guard let nr = extractNormFaceRect(from: cg) else { return nil }
        return CGRect(
            x: nr.minX * viewSize.width,
            y: nr.minY * viewSize.height,
            width: nr.width * viewSize.width,
            height: nr.height * viewSize.height
        )
    }

    //MARK: - Animations

    private func startAnimations() {
        //Glitch
        withAnimation(.easeInOut(duration: 0.07).repeatForever(autoreverses: true)) {
            glitchX = CGFloat.random(in: -2...2)
        }
        //Hue shift
        withAnimation(.easeInOut(duration: 3.0).repeatForever(autoreverses: true)) {
            hueShift = 12
        }
        //Vignette pulse
        withAnimation(.easeInOut(duration: 2.0).repeatForever(autoreverses: true)) {
            vignettePulse = 0.45
        }
        //Edge glow
        withAnimation(.easeInOut(duration: 1.5).repeatForever(autoreverses: true)) {
            edgeGlow = 0.5
        }
    }
}

#if DEBUG
struct EncryptionProgressView_Previews: PreviewProvider {
    static var previews: some View {
        let job = EncryptionJob(
            image: NSImage(systemSymbolName: "photo", accessibilityDescription: nil) ?? NSImage(),
            iterations: 200
        )
        job.status = .protecting
        job.updateProgress(iteration: 75)
        job.startTime = Date().addingTimeInterval(-30)
        return EncryptionProgressView(job: job, onCancel: {})
    }
}
#endif
