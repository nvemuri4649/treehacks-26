//
// CenaLogo.swift
// Cena
//
// 3x3 grid logo with merged tiles — always gently animating
//

import SwiftUI

struct CenaLogo: View {
    let size: CGFloat
    var isAnimating: Bool = false
    var color: Color = .white

    //3x3 grid layout:
    // Row 0: [1x1] [  2x1  ]
    // Row 1: [1x1] [1x1] [1x1]
    // Row 2: [  2x1  ] [1x1]
    private static let tiles: [(col: Int, row: Int, w: Int, h: Int)] = [
        (0, 0, 1, 1), (1, 0, 2, 1),
        (0, 1, 1, 1), (1, 1, 1, 1), (2, 1, 1, 1),
        (0, 2, 2, 1), (2, 2, 1, 1),
    ]

    //Continuous gentle floating offsets per tile (always active)
    private static let floatAngles: [Double] = [0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8]

    @State private var breathe: CGFloat = 0
    @State private var phase: Int = 0
    @State private var slideTimer: Timer?

    //Perimeter for active slide animation
    private static let perimeterOrder: [Int] = [0, 1, 4, 6, 5, 2]

    var body: some View {
        let gridCols = 3
        let gap: CGFloat = size * 0.1
        let step = (size - gap * CGFloat(gridCols - 1)) / CGFloat(gridCols)
        let cornerR = step * 0.3
        let drift: CGFloat = size * 0.02 // gentle drift amount

        Canvas { context, _ in
            for (index, tile) in Self.tiles.enumerated() {
                //Gentle continuous float
                let angle = Self.floatAngles[index] + Double(breathe)
                let fx = CGFloat(sin(angle * 2.1)) * drift
                let fy = CGFloat(cos(angle * 1.7)) * drift

                //Active slide offset (only when processing)
                let slide = slideOffset(for: index, step: step, gap: gap)

                let w = CGFloat(tile.w) * step + CGFloat(tile.w - 1) * gap
                let h = CGFloat(tile.h) * step + CGFloat(tile.h - 1) * gap
                let x = CGFloat(tile.col) * (step + gap) + fx + slide.width
                let y = CGFloat(tile.row) * (step + gap) + fy + slide.height

                let rect = CGRect(x: x, y: y, width: w, height: h)
                let path = RoundedRectangle(cornerRadius: cornerR, style: .continuous)
                    .path(in: rect)
                context.fill(path, with: .color(color))
            }
        }
        .frame(width: size, height: size)
        .onAppear { startBreathing() }
        .onDisappear { stopAll() }
        .onChange(of: isAnimating) { _, on in
            if on { startSliding() } else { stopSliding() }
        }
    }

    //MARK: - Slide (active processing only)

    private func slideOffset(for index: Int, step: CGFloat, gap: CGFloat) -> CGSize {
        guard isAnimating, phase > 0 else { return .zero }
        guard let pi = Self.perimeterOrder.firstIndex(of: index) else { return .zero }

        let count = Self.perimeterOrder.count
        let shift = phase % count
        if shift == 0 { return .zero }

        let targetIdx = Self.perimeterOrder[(pi + shift) % count]
        let src = Self.tiles[index]
        let dst = Self.tiles[targetIdx]

        return CGSize(
            width: CGFloat(dst.col - src.col) * (step + gap),
            height: CGFloat(dst.row - src.row) * (step + gap)
        )
    }

    //MARK: - Continuous breathing (always on)

    private func startBreathing() {
        withAnimation(.linear(duration: 4).repeatForever(autoreverses: false)) {
            breathe = .pi * 2
        }
        if isAnimating { startSliding() }
    }

    private func startSliding() {
        stopSliding()
        phase = 0
        let t = Timer(timeInterval: 0.5, repeats: true) { _ in
            Task { @MainActor in
                withAnimation(.easeInOut(duration: 0.4)) { phase += 1 }
            }
        }
        RunLoop.main.add(t, forMode: .common)
        slideTimer = t
    }

    private func stopSliding() {
        slideTimer?.invalidate()
        slideTimer = nil
        withAnimation(.easeInOut(duration: 0.2)) { phase = 0 }
    }

    private func stopAll() {
        stopSliding()
    }
}
