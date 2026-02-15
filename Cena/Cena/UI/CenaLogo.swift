//
//  CenaLogo.swift
//  Cena
//
//  4x4 grid logo with merged tiles and sliding animation
//

import SwiftUI

struct CenaLogo: View {
    let size: CGFloat
    var isAnimating: Bool = false

    private let accentGrad = LinearGradient(
        colors: [.blue, .purple],
        startPoint: .topLeading, endPoint: .bottomTrailing
    )

    // Each tile: (col, row, colSpan, rowSpan)
    private static let baseTiles: [(col: Int, row: Int, w: Int, h: Int)] = [
        // Row 0: single, merged 2x1, single
        (0, 0, 1, 1), (1, 0, 2, 1), (3, 0, 1, 1),
        // Row 1: merged 2x1, single, single
        (0, 1, 2, 1), (2, 1, 1, 1), (3, 1, 1, 1),
        // Row 2: single, single, merged 2x1
        (0, 2, 1, 1), (1, 2, 1, 1), (2, 2, 2, 1),
        // Row 3: four singles
        (0, 3, 1, 1), (1, 3, 1, 1), (2, 3, 1, 1), (3, 3, 1, 1),
    ]

    // Perimeter indices in clockwise order for animation
    // These are the indices into baseTiles that sit on the outer ring
    // Row0: 0,1,2 | Right col: 5,8,12 (idx) | Bottom: 12,11,10,9 | Left: 6,3,0
    // Outer ring tile indices: 0,1,2, 5,8, 12,11,10,9, 6,3
    // Inner tiles: 4,7 (row1-col2, row2-col1)
    private static let perimeterOrder: [Int] = [0, 1, 2, 5, 8, 12, 11, 10, 9, 6, 3]

    @State private var phase: Int = 0
    @State private var timer: Timer?

    var body: some View {
        let gridSize = 4
        let gap: CGFloat = size * 0.08
        let step = (size - gap * CGFloat(gridSize - 1)) / CGFloat(gridSize)
        let cornerR = step * 0.28

        ZStack(alignment: .topLeading) {
            ForEach(Array(Self.baseTiles.enumerated()), id: \.offset) { index, tile in
                let offset = animationOffset(for: index, step: step, gap: gap)

                RoundedRectangle(cornerRadius: cornerR, style: .continuous)
                    .fill(accentGrad)
                    .frame(
                        width: CGFloat(tile.w) * step + CGFloat(tile.w - 1) * gap,
                        height: CGFloat(tile.h) * step + CGFloat(tile.h - 1) * gap
                    )
                    .offset(
                        x: CGFloat(tile.col) * (step + gap) + offset.width,
                        y: CGFloat(tile.row) * (step + gap) + offset.height
                    )
            }
        }
        .frame(width: size, height: size)
        .onAppear { startIfNeeded() }
        .onDisappear { stopTimer() }
        .onChange(of: isAnimating) { _, animating in
            if animating { startTimer() } else { stopTimer() }
        }
    }

    // MARK: - Animation

    private func animationOffset(for index: Int, step: CGFloat, gap: CGFloat) -> CGSize {
        guard isAnimating else { return .zero }

        // Find this tile's position in the perimeter ring
        guard let perimIdx = Self.perimeterOrder.firstIndex(of: index) else {
            return .zero // inner tiles don't move
        }

        let count = Self.perimeterOrder.count
        // How many steps this tile has shifted from its home
        let shift = phase % count

        if shift == 0 { return .zero }

        // Compute where this tile should currently be
        let targetPerimIdx = (perimIdx + shift) % count
        let targetTileIdx = Self.perimeterOrder[targetPerimIdx]

        let src = Self.baseTiles[index]
        let dst = Self.baseTiles[targetTileIdx]

        let dx = CGFloat(dst.col - src.col) * (step + gap)
        let dy = CGFloat(dst.row - src.row) * (step + gap)

        return CGSize(width: dx, height: dy)
    }

    private func startIfNeeded() {
        if isAnimating { startTimer() }
    }

    private func startTimer() {
        stopTimer()
        phase = 0
        timer = Timer.scheduledTimer(withTimeInterval: 0.45, repeats: true) { _ in
            Task { @MainActor in
                withAnimation(.easeInOut(duration: 0.35)) {
                    phase += 1
                }
            }
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
        withAnimation(.easeInOut(duration: 0.2)) {
            phase = 0
        }
    }
}
