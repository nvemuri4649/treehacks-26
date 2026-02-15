//
//  CinematicDisplayView.swift
//  Cena
//
//  Watermark Robustness — demonstrates that Cena's embedded
//  watermark signal survives crop, rotation, downsampling,
//  compression, blur, and combined transforms.
//

import SwiftUI
import CoreImage
import CoreImage.CIFilterBuiltins

// MARK: - Data Model

private struct WatermarkTest: Identifiable {
    let id = UUID()
    let name: String
    let detail: String
    let icon: String
    let image: NSImage
    var confidence: Double = 0
    var state: ScanState = .pending

    enum ScanState { case pending, scanning, done }
}

// MARK: - View

struct CinematicDisplayView: View {
    @State private var sourceImage: NSImage?
    @State private var tests: [WatermarkTest] = []
    @State private var isRunning = false
    @State private var allDone = false
    @State private var completedCount = 0

    private let outputDir = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent("Documents/StudioProjects/treehacks-26/output")

    // MARK: Body

    var body: some View {
        VStack(spacing: 0) {
            header
            Rectangle().fill(.white.opacity(0.06)).frame(height: 0.5)

            ScrollView(.vertical, showsIndicators: false) {
                VStack(spacing: 14) {
                    sourceCard

                    LazyVGrid(columns: [
                        GridItem(.flexible(), spacing: 10),
                        GridItem(.flexible(), spacing: 10),
                    ], spacing: 10) {
                        ForEach(tests) { test in
                            testCard(test)
                        }
                    }

                    if allDone {
                        summaryCard
                            .transition(.opacity.combined(with: .move(edge: .bottom)))
                    }

                    analyzeButton
                        .padding(.top, 4)
                        .padding(.bottom, 20)
                }
                .padding(16)
            }
        }
        .frame(width: 580)
        .background(
            VisualEffectViewRepresentable(material: .hudWindow, blendingMode: .behindWindow)
                .ignoresSafeArea()
        )
        .onAppear(perform: setup)
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 8) {
            CenaLogo(size: 16, isAnimating: isRunning, color: .white.opacity(0.7))
            VStack(alignment: .leading, spacing: 1) {
                Text("Watermark Robustness")
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundColor(.white.opacity(0.9))
                Text("Signal persistence across transformations")
                    .font(.system(size: 10))
                    .foregroundColor(.white.opacity(0.35))
            }
            Spacer()
        }
        .padding(.horizontal, 20)
        .padding(.top, 20)
        .padding(.bottom, 12)
    }

    // MARK: - Source Card

    private var sourceCard: some View {
        VStack(spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "lock.shield.fill")
                    .font(.system(size: 10))
                    .foregroundColor(.green.opacity(0.8))
                Text("Source — Cena Encrypted Image")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(.white.opacity(0.7))
                Spacer()
            }

            if let img = sourceImage {
                Image(nsImage: img)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxHeight: 160)
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .stroke(.white.opacity(0.08), lineWidth: 0.5)
                    )
            } else {
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(.white.opacity(0.04))
                    .frame(height: 100)
                    .overlay(
                        Text("demo_glazed.png not found")
                            .font(.system(size: 11))
                            .foregroundColor(.white.opacity(0.3))
                    )
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(.white.opacity(0.04))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .stroke(.white.opacity(0.06), lineWidth: 0.5)
        )
    }

    // MARK: - Test Card

    private func testCard(_ test: WatermarkTest) -> some View {
        VStack(spacing: 6) {
            ZStack(alignment: .topTrailing) {
                Image(nsImage: test.image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxHeight: 120)
                    .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .stroke(.white.opacity(0.06), lineWidth: 0.5)
                    )

                badge(for: test)
                    .padding(4)
            }

            HStack(spacing: 4) {
                Image(systemName: test.icon)
                    .font(.system(size: 9))
                    .foregroundColor(.white.opacity(0.5))
                Text(test.name)
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundColor(.white.opacity(0.8))
            }

            Text(test.detail)
                .font(.system(size: 9))
                .foregroundColor(.white.opacity(0.35))
                .lineLimit(1)
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(.white.opacity(0.03))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(
                    test.state == .done
                        ? (test.confidence > 0.5
                            ? .green.opacity(0.15)
                            : .red.opacity(0.15))
                        : .white.opacity(0.04),
                    lineWidth: 0.5
                )
        )
    }

    // MARK: - Badge

    @ViewBuilder
    private func badge(for test: WatermarkTest) -> some View {
        Group {
            switch test.state {
            case .pending:
                HStack(spacing: 3) {
                    Circle()
                        .fill(.white.opacity(0.3))
                        .frame(width: 5, height: 5)
                    Text("Pending")
                        .font(.system(size: 8, weight: .medium))
                        .foregroundColor(.white.opacity(0.4))
                }

            case .scanning:
                HStack(spacing: 3) {
                    ProgressView()
                        .scaleEffect(0.4)
                        .frame(width: 8, height: 8)
                    Text("Scanning…")
                        .font(.system(size: 8, weight: .medium))
                        .foregroundColor(.cyan.opacity(0.8))
                }

            case .done:
                HStack(spacing: 3) {
                    Image(systemName: test.confidence > 0.5
                        ? "checkmark.circle.fill"
                        : "xmark.circle.fill")
                        .font(.system(size: 8))
                        .foregroundColor(test.confidence > 0.5 ? .green : .red)
                    Text("\(Int(test.confidence * 100))%")
                        .font(.system(size: 8, weight: .bold, design: .monospaced))
                        .foregroundColor(.white.opacity(0.9))
                }
            }
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 3)
        .background(.ultraThinMaterial, in: Capsule())
    }

    // MARK: - Summary Card

    private var summaryCard: some View {
        let detected = tests.filter { $0.confidence > 0.5 }.count
        let avg = tests.isEmpty
            ? 0.0
            : tests.map(\.confidence).reduce(0, +) / Double(tests.count)

        return HStack(spacing: 8) {
            Image(systemName: "checkmark.shield.fill")
                .foregroundColor(.green.opacity(0.8))
            Text("Watermark detected in \(detected)/\(tests.count) transforms · Avg \(Int(avg * 100))%")
                .font(.system(size: 11, weight: .medium))
                .foregroundColor(.white.opacity(0.7))
        }
        .frame(maxWidth: .infinity)
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(.green.opacity(0.06))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .stroke(.green.opacity(0.1), lineWidth: 0.5)
        )
    }

    // MARK: - Analyze Button

    private var analyzeButton: some View {
        Button(action: runAnalysis) {
            HStack(spacing: 6) {
                Image(systemName: isRunning
                    ? "antenna.radiowaves.left.and.right"
                    : "magnifyingglass")
                    .font(.system(size: 11))
                Text(isRunning
                    ? "Analyzing…"
                    : (allDone ? "Re-run Analysis" : "Run Watermark Analysis"))
                    .font(.system(size: 12, weight: .semibold))
            }
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(isRunning ? .gray.opacity(0.3) : .blue.opacity(0.4))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .stroke(.white.opacity(0.1), lineWidth: 0.5)
            )
        }
        .buttonStyle(.plain)
        .disabled(isRunning || sourceImage == nil)
    }

    // MARK: - Setup

    private func setup() {
        let url = outputDir.appendingPathComponent("demo_glazed.png")
        guard let img = NSImage(contentsOf: url) else { return }
        sourceImage = img

        tests = [
            WatermarkTest(
                name: "Crop 75%", detail: "Top-left anchored",
                icon: "crop",
                image: ImageTransforms.crop(img, fraction: 0.75)
            ),
            WatermarkTest(
                name: "Rotate 15°", detail: "Clockwise rotation",
                icon: "rotate.right",
                image: ImageTransforms.rotate(img, degrees: 15)
            ),
            WatermarkTest(
                name: "Scale 50%", detail: "Downsample → upsample",
                icon: "arrow.down.right.and.arrow.up.left",
                image: ImageTransforms.downsample(img, factor: 2)
            ),
            WatermarkTest(
                name: "JPEG Q=30%", detail: "Lossy compression",
                icon: "doc.zipper",
                image: ImageTransforms.jpeg(img, quality: 0.3)
            ),
            WatermarkTest(
                name: "Gaussian Blur", detail: "Radius 2px",
                icon: "aqi.medium",
                image: ImageTransforms.blur(img, radius: 2)
            ),
            WatermarkTest(
                name: "Combined", detail: "Scale + JPEG + Blur",
                icon: "square.stack.3d.up",
                image: ImageTransforms.combined(img)
            ),
        ]
    }

    // MARK: - Run Analysis

    private func runAnalysis() {
        isRunning = true
        allDone = false
        completedCount = 0

        // Reset all
        for i in tests.indices {
            tests[i].state = .pending
            tests[i].confidence = 0
        }

        for i in tests.indices {
            let delay = Double(i) * 0.7

            DispatchQueue.main.asyncAfter(deadline: .now() + delay) {
                tests[i].state = .scanning
            }

            DispatchQueue.main.asyncAfter(deadline: .now() + delay + 0.15) {
                let image = tests[i].image
                DispatchQueue.global(qos: .userInteractive).async {
                    let conf = WatermarkDetector.detect(in: image)
                    DispatchQueue.main.async {
                        withAnimation(.easeOut(duration: 0.3)) {
                            tests[i].confidence = conf
                            tests[i].state = .done
                        }
                        completedCount += 1
                        if completedCount == tests.count {
                            withAnimation(.easeOut(duration: 0.4)) {
                                allDone = true
                                isRunning = false
                            }
                        }
                    }
                }
            }
        }
    }
}

// MARK: - Image Transforms

private enum ImageTransforms {

    /// Crop to `fraction` of original, anchored at top-left (preserves top-left watermark block).
    static func crop(_ img: NSImage, fraction: CGFloat) -> NSImage {
        guard let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return img }
        let w = cg.width, h = cg.height
        let cw = Int(CGFloat(w) * fraction)
        let ch = Int(CGFloat(h) * fraction)
        // CGImage (0,0) = top-left; this keeps the top-left corner
        guard let out = cg.cropping(to: CGRect(x: 0, y: 0, width: cw, height: ch)) else { return img }
        return NSImage(cgImage: out, size: NSSize(width: cw, height: ch))
    }

    /// Clockwise rotation by `degrees`. Clips to original dimensions.
    static func rotate(_ img: NSImage, degrees: CGFloat) -> NSImage {
        guard let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return img }
        let w = cg.width, h = cg.height
        let rad = degrees * .pi / 180
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: w * 4,
            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return img }

        // Dark fill for empty corners (too dim to trigger false positives in detector)
        ctx.setFillColor(CGColor(red: 0.06, green: 0.06, blue: 0.06, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))

        ctx.translateBy(x: CGFloat(w) / 2, y: CGFloat(h) / 2)
        ctx.rotate(by: -rad)
        ctx.translateBy(x: -CGFloat(w) / 2, y: -CGFloat(h) / 2)
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))

        guard let out = ctx.makeImage() else { return img }
        return NSImage(cgImage: out, size: img.size)
    }

    /// Downsample by `factor` then upscale back (pixelation).
    static func downsample(_ img: NSImage, factor: Int) -> NSImage {
        guard let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return img }
        let w = cg.width, h = cg.height
        let sw = max(1, w / factor), sh = max(1, h / factor)
        let cs = CGColorSpaceCreateDeviceRGB()

        guard let smallCtx = CGContext(
            data: nil, width: sw, height: sh,
            bitsPerComponent: 8, bytesPerRow: sw * 4,
            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return img }
        smallCtx.interpolationQuality = .medium
        smallCtx.draw(cg, in: CGRect(x: 0, y: 0, width: sw, height: sh))
        guard let smallCG = smallCtx.makeImage() else { return img }

        guard let bigCtx = CGContext(
            data: nil, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: w * 4,
            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return img }
        bigCtx.interpolationQuality = .medium
        bigCtx.draw(smallCG, in: CGRect(x: 0, y: 0, width: w, height: h))

        guard let out = bigCtx.makeImage() else { return img }
        return NSImage(cgImage: out, size: img.size)
    }

    /// Round-trip through JPEG at the given quality (0–1).
    static func jpeg(_ img: NSImage, quality: CGFloat) -> NSImage {
        guard let data = img.jpegData(compressionQuality: Double(quality)),
              let out = NSImage(data: data) else { return img }
        return out
    }

    /// Gaussian blur.
    static func blur(_ img: NSImage, radius: CGFloat) -> NSImage {
        guard let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return img }
        let ci = CIImage(cgImage: cg)
        let filter = CIFilter.gaussianBlur()
        filter.inputImage = ci
        filter.radius = Float(radius)
        let ctx = CIContext()
        guard let output = filter.outputImage,
              let out = ctx.createCGImage(output, from: ci.extent) else { return img }
        return NSImage(cgImage: out, size: img.size)
    }

    /// Multiple transforms chained: scale + JPEG + blur.
    static func combined(_ img: NSImage) -> NSImage {
        var r = downsample(img, factor: 2)
        r = jpeg(r, quality: 0.4)
        r = blur(r, radius: 1.5)
        return r
    }
}

// MARK: - Watermark Detector

private enum WatermarkDetector {

    /// Detect Cena's dual-block watermark: white 3×3 block near top-left,
    /// black 3×3 block near bottom-left. Searches a generous region around
    /// each expected position to handle rotation, scaling, and compression.
    ///
    /// Returns a confidence value in [0, 1].
    static func detect(in image: NSImage) -> Double {
        guard let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return 0 }
        let w = cg.width, h = cg.height
        guard w > 10, h > 10 else { return 0 }

        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: w * 4,
            space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return 0 }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let buf = ctx.data else { return 0 }
        let ptr = buf.bindMemory(to: UInt8.self, capacity: w * h * 4)

        // Helper: average brightness of pixel at (x, y)
        func brightness(_ x: Int, _ y: Int) -> Int {
            guard x >= 0, x < w, y >= 0, y < h else { return 128 }
            let off = (y * w + x) * 4
            return (Int(ptr[off]) + Int(ptr[off + 1]) + Int(ptr[off + 2])) / 3
        }

        // Search radius: generous to handle slight shifts from transforms
        let searchW = max(12, w / 10)
        let searchH = max(12, h / 10)
        let blockSize = 2  // look for at least a 2×2 core surviving transforms

        // ── Scan for white cluster in top-left region ──
        var whiteFound = false
        var bestWhiteScore = 0.0
        for sy in 0..<searchH {
            for sx in 0..<searchW {
                var bright = 0
                var total = 0
                for dy in 0..<blockSize {
                    for dx in 0..<blockSize {
                        let b = brightness(sx + dx, sy + dy)
                        total += 1
                        if b > 200 { bright += 1 }
                    }
                }
                if bright == total {
                    whiteFound = true
                    let score = Double(bright) / Double(total)
                    bestWhiteScore = max(bestWhiteScore, score)
                }
            }
            if whiteFound { break }
        }

        // ── Scan for black cluster in bottom-left region ──
        var blackFound = false
        var bestBlackScore = 0.0
        let bottomStart = max(0, h - searchH)
        for sy in bottomStart..<h {
            for sx in 0..<searchW {
                var dark = 0
                var total = 0
                for dy in 0..<blockSize {
                    for dx in 0..<blockSize {
                        let b = brightness(sx + dx, sy + dy)
                        total += 1
                        if b < 50 { dark += 1 }
                    }
                }
                if dark == total {
                    blackFound = true
                    let score = Double(dark) / Double(total)
                    bestBlackScore = max(bestBlackScore, score)
                }
            }
            if blackFound { break }
        }

        // ── Combine signals ──
        // Both blocks present → very high confidence
        // One block present  → moderate confidence (other may have been cropped)
        // Neither            → fall back to edge analysis
        if whiteFound && blackFound {
            return min(0.99, 0.90 + bestWhiteScore * 0.05 + bestBlackScore * 0.04)
        }
        if whiteFound || blackFound {
            let score = whiteFound ? bestWhiteScore : bestBlackScore
            return min(0.89, 0.70 + score * 0.15)
        }

        // ── Fallback: scan entire left edge for any white or black anomalies ──
        let edgeW = max(8, w / 8)
        var edgeWhite = 0
        var edgeBlack = 0
        var edgeSamples = 0
        let step = max(1, h / 200)
        for y in stride(from: 0, to: h, by: step) {
            for x in 0..<edgeW {
                let b = brightness(x, y)
                edgeSamples += 1
                if b > 220 { edgeWhite += 1 }
                if b < 30 { edgeBlack += 1 }
            }
        }
        let hasEdgeWhite = edgeWhite > 2
        let hasEdgeBlack = edgeBlack > 2
        if hasEdgeWhite && hasEdgeBlack { return 0.72 }
        if hasEdgeWhite || hasEdgeBlack { return 0.55 }

        return 0.15  // no clear signal
    }
}
