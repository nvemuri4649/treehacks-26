//
// DemoView.swift
// Cena
//
// Interactive demo: Image encryption + Video deepfake prevention
//

import SwiftUI
import AVFoundation
import AppKit

struct DemoView: View {
    @State private var selectedTab = 0  // 0 = Image, 1 = Video

    var body: some View {
        VStack(spacing: 0) {
            header
            Rectangle().fill(.white.opacity(0.06)).frame(height: 0.5)

            //Tab picker
            Picker("", selection: $selectedTab) {
                Text("Image Demo").tag(0)
                Text("Video Demo").tag(1)
            }
            .pickerStyle(.segmented)
            .padding(.horizontal, 28)
            .padding(.top, 14)
            .padding(.bottom, 6)

            if selectedTab == 0 {
                ImageDemoTab()
            } else {
                VideoDemoTab()
            }
        }
        .frame(minWidth: 860, minHeight: 680)
        .background(
            VisualEffectViewRepresentable(material: .hudWindow, blendingMode: .behindWindow)
                .ignoresSafeArea()
        )
    }

    private var header: some View {
        HStack(spacing: 10) {
            CenaLogo(size: 22, isAnimating: false, color: .white.opacity(0.8))
            Text("Likeness Encryption Demo")
                .font(.system(size: 15, weight: .semibold, design: .rounded))
            Spacer()
        }
        .padding(.horizontal, 28)
        .padding(.vertical, 14)
        .padding(.top, 12)
    }
}

//MARK: - Shared

private let demoDir: URL? = {
    let candidates = [
        URL(fileURLWithPath: NSHomeDirectory())
            .appendingPathComponent("Documents/StudioProjects/treehacks-26/output"),
        Bundle.main.resourceURL,
    ].compactMap { $0 }
    for dir in candidates {
        if FileManager.default.fileExists(atPath: dir.appendingPathComponent("demo_original.png").path) {
            return dir
        }
    }
    return candidates.first
}()

//MARK: - Image Demo Tab

struct ImageDemoTab: View {
    @State private var showDiffused = false
    @State private var isLoading = false
    @State private var loadProgress: CGFloat = 0

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                inputSection
                buttonRow
                if isLoading { progressBar }
                if showDiffused {
                    outputSection
                        .transition(.opacity.combined(with: .move(edge: .bottom)))
                }
                explanationCard
            }
            .padding(28)
        }
    }

    private var inputSection: some View {
        VStack(spacing: 8) {
            sectionHeader("INPUT IMAGES", subtitle: "Nearly identical — can you tell the difference?")
            HStack(spacing: 14) {
                DemoImageCard(filename: "demo_original.png", label: "Original", sublabel: "Unprotected", badgeColor: .red)
                DemoImageCard(filename: "demo_glazed.png", label: "Encrypted", sublabel: "Cena Protected", badgeColor: .green)
            }
        }
    }

    private var buttonRow: some View {
        Button {
            guard !isLoading, !showDiffused else {
                withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) { showDiffused = false; isLoading = false; loadProgress = 0 }
                return
            }
            isLoading = true; loadProgress = 0
            simulateProgress()
        } label: {
            HStack(spacing: 8) {
                Image(systemName: showDiffused ? "eye.slash" : "wand.and.stars")
                    .font(.system(size: 12))
                Text(showDiffused ? "Hide Results" : "Reveal Both Post-AI Transformation")
                    .font(.system(size: 12, weight: .semibold))
            }
            .foregroundColor(.white)
            .padding(.horizontal, 24)
            .padding(.vertical, 10)
            .background(
                LinearGradient(colors: [.blue, .purple], startPoint: .leading, endPoint: .trailing).opacity(0.7),
                in: Capsule()
            )
            .overlay(Capsule().stroke(.white.opacity(0.1), lineWidth: 0.5))
        }
        .buttonStyle(.plain)
    }

    private var progressBar: some View {
        VStack(spacing: 6) {
            HStack {
                Text("Running diffusion model...")
                    .font(.system(size: 10))
                    .foregroundColor(.white.opacity(0.4))
                Spacer()
                Text("\(Int(loadProgress * 100))%")
                    .font(.system(size: 10, weight: .bold, design: .rounded))
                    .foregroundColor(.white.opacity(0.5))
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(.white.opacity(0.08))
                        .frame(height: 6)
                    RoundedRectangle(cornerRadius: 4)
                        .fill(LinearGradient(colors: [.blue, .purple], startPoint: .leading, endPoint: .trailing))
                        .frame(width: geo.size.width * loadProgress, height: 6)
                        .animation(.easeInOut(duration: 0.3), value: loadProgress)
                }
            }
            .frame(height: 6)
        }
        .padding(.horizontal, 4)
    }

    private var outputSection: some View {
        VStack(spacing: 8) {
            sectionHeader("AI GENERATION OUTPUT", subtitle: "Same prompt applied to both images")
            HStack(spacing: 14) {
                DemoImageCard(filename: "demo_original_diffused.png", label: "Unprotected Result", sublabel: "Deepfake succeeded", badgeColor: .red)
                DemoImageCard(filename: "demo_glazed_diffused.png", label: "Protected Result", sublabel: "Deepfake defeated", badgeColor: .green)
            }
        }
    }

    private func simulateProgress() {
        let steps = 30
        for i in 1...steps {
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(i) * 0.12) {
                loadProgress = CGFloat(i) / CGFloat(steps)
                if i == steps {
                    isLoading = false
                    withAnimation(.spring(response: 0.5, dampingFraction: 0.8)) { showDiffused = true }
                }
            }
        }
    }
}

//MARK: - Video Demo Tab

struct VideoDemoTab: View {
    @State private var showResult = false
    @State private var isLoading = false
    @State private var loadProgress: CGFloat = 0

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                inputSection
                buttonRow
                if isLoading { progressBar }
                if showResult {
                    outputSection
                        .transition(.opacity.combined(with: .move(edge: .bottom)))
                }
                explanationCard
            }
            .padding(28)
        }
    }

    private var inputSection: some View {
        VStack(spacing: 8) {
            sectionHeader("INPUT IMAGES", subtitle: "Starting frames — both videos begin with this face")
            HStack(spacing: 14) {
                DemoImageCard(filename: "demo_original.png", label: "Original Photo", sublabel: "Unprotected", badgeColor: .red)
                DemoImageCard(filename: "demo_glazed.png", label: "Encrypted Photo", sublabel: "Cena Protected", badgeColor: .green)
            }
        }
    }

    private var buttonRow: some View {
        Button {
            guard !isLoading, !showResult else {
                withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) { showResult = false; isLoading = false; loadProgress = 0 }
                return
            }
            isLoading = true; loadProgress = 0
            simulateProgress()
        } label: {
            HStack(spacing: 8) {
                Image(systemName: showResult ? "eye.slash" : "video.fill")
                    .font(.system(size: 12))
                Text(showResult ? "Hide Results" : "Reveal Both Post-AI Video Generation")
                    .font(.system(size: 12, weight: .semibold))
            }
            .foregroundColor(.white)
            .padding(.horizontal, 24)
            .padding(.vertical, 10)
            .background(
                LinearGradient(colors: [.blue, .purple], startPoint: .leading, endPoint: .trailing).opacity(0.7),
                in: Capsule()
            )
            .overlay(Capsule().stroke(.white.opacity(0.1), lineWidth: 0.5))
        }
        .buttonStyle(.plain)
    }

    private var progressBar: some View {
        VStack(spacing: 6) {
            HStack {
                Text("Generating video with AI model...")
                    .font(.system(size: 10))
                    .foregroundColor(.white.opacity(0.4))
                Spacer()
                Text("\(Int(loadProgress * 100))%")
                    .font(.system(size: 10, weight: .bold, design: .rounded))
                    .foregroundColor(.white.opacity(0.5))
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(.white.opacity(0.08))
                        .frame(height: 6)
                    RoundedRectangle(cornerRadius: 4)
                        .fill(LinearGradient(colors: [.blue, .purple], startPoint: .leading, endPoint: .trailing))
                        .frame(width: geo.size.width * loadProgress, height: 6)
                        .animation(.easeInOut(duration: 0.3), value: loadProgress)
                }
            }
            .frame(height: 6)
        }
        .padding(.horizontal, 4)
    }

    private var outputSection: some View {
        VStack(spacing: 8) {
            sectionHeader("AI VIDEO GENERATION OUTPUT", subtitle: "Same generation prompt applied to both")
            HStack(spacing: 14) {
                videoCard(filename: "demo_original_video.mp4", label: "Unprotected Result", sublabel: "Deepfake succeeded", badgeColor: .red)
                videoCard(filename: "demo_glazed_video.mp4", label: "Protected Result", sublabel: "Deepfake defeated", badgeColor: .green)
            }
        }
    }

    private func videoCard(filename: String, label: String, sublabel: String, badgeColor: Color) -> some View {
        VStack(spacing: 8) {
            ZStack {
                if let url = demoDir?.appendingPathComponent(filename),
                   FileManager.default.fileExists(atPath: url.path) {
                    LoopingVideoView(url: url)
                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .stroke(.white.opacity(0.08), lineWidth: 0.5)
                        )
                        .aspectRatio(1, contentMode: .fit)
                } else {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .fill(.white.opacity(0.04))
                        .aspectRatio(1, contentMode: .fit)
                        .overlay(
                            VStack(spacing: 6) {
                                Image(systemName: "video.slash")
                                    .font(.system(size: 24))
                                    .foregroundColor(.white.opacity(0.15))
                                Text(filename)
                                    .font(.system(size: 9, design: .monospaced))
                                    .foregroundColor(.white.opacity(0.2))
                            }
                        )
                }

                VStack {
                    HStack {
                        HStack(spacing: 4) {
                            Circle().fill(badgeColor).frame(width: 6, height: 6)
                            Text(sublabel)
                                .font(.system(size: 9, weight: .semibold))
                                .foregroundColor(.white.opacity(0.9))
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(.ultraThinMaterial, in: Capsule())
                        .overlay(Capsule().stroke(.white.opacity(0.1), lineWidth: 0.5))
                        Spacer()
                    }
                    .padding(8)
                    Spacer()
                }
            }

            Text(label)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(.white.opacity(0.5))
        }
    }

    private func simulateProgress() {
        let steps = 50
        for i in 1...steps {
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(i) * 0.14) {
                loadProgress = CGFloat(i) / CGFloat(steps)
                if i == steps {
                    isLoading = false
                    withAnimation(.spring(response: 0.5, dampingFraction: 0.8)) { showResult = true }
                }
            }
        }
    }
}

//MARK: - Shared Components

private func sectionHeader(_ title: String, subtitle: String) -> some View {
    HStack(spacing: 5) {
        Text(title)
            .font(.system(size: 9, weight: .bold))
            .foregroundColor(.white.opacity(0.3))
            .tracking(1.5)
        Spacer()
        Text(subtitle)
            .font(.system(size: 10))
            .foregroundColor(.white.opacity(0.25))
    }
}

private var explanationCard: some View {
    VStack(alignment: .leading, spacing: 10) {
        HStack(spacing: 6) {
            Image(systemName: "info.circle")
                .font(.system(size: 11))
                .foregroundColor(.blue.opacity(0.7))
            Text("How It Works")
                .font(.system(size: 11, weight: .semibold))
                .foregroundColor(.white.opacity(0.6))
        }
        Text("Cena applies imperceptible adversarial perturbations to encrypt your media. These are invisible to the human eye but cause AI models to produce corrupted outputs when attempting deepfakes. The encrypted media looks identical to the original but is protected against malicious AI manipulation.")
            .font(.system(size: 11))
            .foregroundColor(.white.opacity(0.35))
            .lineSpacing(3)
            .fixedSize(horizontal: false, vertical: true)
    }
    .padding(16)
    .background(.white.opacity(0.03), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    .overlay(
        RoundedRectangle(cornerRadius: 12, style: .continuous)
            .stroke(.white.opacity(0.05), lineWidth: 0.5)
    )
}

struct DemoImageCard: View {
    let filename: String
    let label: String
    let sublabel: String
    let badgeColor: Color

    var body: some View {
        VStack(spacing: 8) {
            ZStack {
                if let url = demoDir?.appendingPathComponent(filename),
                   let image = NSImage(contentsOf: url) {
                    Image(nsImage: image)
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
                        .overlay(
                            VStack(spacing: 6) {
                                Image(systemName: "photo")
                                    .font(.system(size: 24))
                                    .foregroundColor(.white.opacity(0.15))
                                Text(filename)
                                    .font(.system(size: 9, design: .monospaced))
                                    .foregroundColor(.white.opacity(0.2))
                            }
                        )
                }

                VStack {
                    HStack {
                        HStack(spacing: 4) {
                            Circle().fill(badgeColor).frame(width: 6, height: 6)
                            Text(sublabel)
                                .font(.system(size: 9, weight: .semibold))
                                .foregroundColor(.white.opacity(0.9))
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(.ultraThinMaterial, in: Capsule())
                        .overlay(Capsule().stroke(.white.opacity(0.1), lineWidth: 0.5))
                        Spacer()
                    }
                    .padding(8)
                    Spacer()
                }
            }

            Text(label)
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(.white.opacity(0.5))
        }
    }
}

//MARK: - Looping video via AVPlayerLayer (no AVKit import)

struct LoopingVideoView: NSViewRepresentable {
    let url: URL

    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        view.wantsLayer = true

        let player = AVPlayer(url: url)
        player.isMuted = true

        let layer = AVPlayerLayer(player: player)
        layer.videoGravity = .resizeAspectFill
        view.layer = layer

        NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime,
            object: player.currentItem,
            queue: .main
        ) { _ in
            player.seek(to: .zero)
            player.play()
        }

        player.play()
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {}
}
