//
//  CenaLogo.swift
//  Cena
//
//  3x3 grid logo with merged tiles and sliding animation
//

import SwiftUI

struct CenaLogo: View {
    let size: CGFloat
    var isAnimating: Bool = false
    var color: Color = .white

    // 3x3 grid layout:
    //  Row 0: [1x1] [  2x1  ]       — 1 merged, then single
    //  Row 1: [1x1] [1x1] [1x1]     — 3 singles
    //  Row 2: [  2x1  ] [1x1]       — merged, then 1 single
    private static let tiles: [(col: Int, row: Int, w: Int, h: Int)] = [
        (0, 0, 1, 1), (1, 0, 2, 1),           // row 0
        (0, 1, 1, 1), (1, 1, 1, 1), (2, 1, 1, 1), // row 1
        (0, 2, 2, 1), (2, 2, 1, 1),           // row 2
    ]

    // Perimeter order (all 7 tiles are on the edge of a 3x3, no inner tile)
    // Clockwise: top-left, top-right-merged, right-mid, bottom-right, bottom-left-merged, left-mid
    // Indices: 0, 1, 4, 6, 5, 2
    private static let perimeterOrder: [Int] = [0, 1, 4, 6, 5, 2]

    @State private var phase: Int = 0
    @State private var timer: Timer?

    var body: some View {
        let gridCols = 3
        let gap: CGFloat = size * 0.1
        let step = (size - gap * CGFloat(gridCols - 1)) / CGFloat(gridCols)
        let cornerR = step * 0.3

        ZStack(alignment: .topLeading) {
            ForEach(Array(Self.tiles.enumerated()), id: \.offset) { index, tile in
                let anim = animOffset(for: index, step: step, gap: gap)

                RoundedRectangle(cornerRadius: cornerR, style: .continuous)
                    .fill(color)
                    .frame(
                        width: CGFloat(tile.w) * step + CGFloat(tile.w - 1) * gap,
                        height: CGFloat(tile.h) * step + CGFloat(tile.h - 1) * gap
                    )
                    .offset(
                        x: CGFloat(tile.col) * (step + gap) + anim.width,
                        y: CGFloat(tile.row) * (step + gap) + anim.height
                    )
            }
        }
        .frame(width: size, height: size)
        .clipped()
        .onAppear { if isAnimating { startTimer() } }
        .onDisappear { stopTimer() }
        .onChange(of: isAnimating) { _, on in
            if on { startTimer() } else { stopTimer() }
        }
    }

    // MARK: - Animation

    private func animOffset(for index: Int, step: CGFloat, gap: CGFloat) -> CGSize {
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

    private func startTimer() {
        stopTimer()
        phase = 0
        timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in
            Task { @MainActor in
                withAnimation(.easeInOut(duration: 0.4)) { phase += 1 }
            }
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
        withAnimation(.easeInOut(duration: 0.2)) { phase = 0 }
    }
}
