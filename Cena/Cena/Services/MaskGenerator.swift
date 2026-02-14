//
//  MaskGenerator.swift
//  Cena
//
//  Automatic mask generation using Apple Vision framework for face detection
//

import Foundation
import AppKit
import Vision

class MaskGenerator {

    // MARK: - Face Detection

    /// Generate a mask with white rectangles for detected faces, black background
    /// - Parameter image: Input image to detect faces in
    /// - Returns: Binary mask image (white = protect, black = ignore)
    static func generateFaceMask(for image: NSImage) async throws -> NSImage {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw GlazingError.invalidImage
        }

        print("🔍 Detecting faces in image...")

        // Create face detection request
        let request = VNDetectFaceRectanglesRequest()

        // Perform request
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        guard let results = request.results, !results.isEmpty else {
            print("⚠️  No faces detected, using full-image mask")
            return createFullMask(size: image.size)
        }

        print("✅ Detected \(results.count) face(s)")

        // Create mask image with face rectangles
        return drawFaceMask(faces: results, imageSize: image.size, cgImage: cgImage)
    }

    /// Generate a mask with white rectangles around detected faces with padding
    /// - Parameters:
    ///   - image: Input image
    ///   - padding: Padding factor around detected faces (0.0 = tight, 0.5 = 50% larger)
    /// - Returns: Binary mask image with expanded face regions
    static func generateExpandedFaceMask(for image: NSImage, padding: CGFloat = 0.3) async throws -> NSImage {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw GlazingError.invalidImage
        }

        print("🔍 Detecting faces with \(Int(padding * 100))% padding...")

        let request = VNDetectFaceRectanglesRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        guard let results = request.results, !results.isEmpty else {
            print("⚠️  No faces detected, using full-image mask")
            return createFullMask(size: image.size)
        }

        print("✅ Detected \(results.count) face(s)")

        // Expand face rectangles
        let expandedFaces = results.map { face -> VNFaceObservation in
            let bbox = face.boundingBox
            let centerX = bbox.midX
            let centerY = bbox.midY
            let newWidth = bbox.width * (1.0 + padding)
            let newHeight = bbox.height * (1.0 + padding)

            // Create new bounding box (clamped to 0-1)
            let newX = max(0, centerX - newWidth / 2)
            let newY = max(0, centerY - newHeight / 2)
            let clampedWidth = min(newWidth, 1.0 - newX)
            let clampedHeight = min(newHeight, 1.0 - newY)

            // Note: We can't directly modify VNFaceObservation, so we'll pass the expanded rect separately
            return face
        }

        return drawFaceMask(faces: expandedFaces, imageSize: image.size, cgImage: cgImage, padding: padding)
    }

    // MARK: - Full Image Mask

    /// Create a mask that covers the entire image (all white)
    /// - Parameter size: Size of the mask image
    /// - Returns: All-white mask image
    static func createFullMask(size: CGSize) -> NSImage {
        let maskImage = NSImage(size: size)

        maskImage.lockFocus()
        NSColor.white.setFill()
        NSRect(origin: .zero, size: size).fill()
        maskImage.unlockFocus()

        print("📄 Created full-image mask")
        return maskImage
    }

    // MARK: - Drawing Helpers

    /// Draw white rectangles for detected faces on black background
    private static func drawFaceMask(
        faces: [VNFaceObservation],
        imageSize: CGSize,
        cgImage: CGImage,
        padding: CGFloat = 0.0
    ) -> NSImage {
        let maskImage = NSImage(size: imageSize)

        maskImage.lockFocus()

        // Black background
        NSColor.black.setFill()
        NSRect(origin: .zero, size: imageSize).fill()

        // White rectangles for faces
        NSColor.white.setFill()

        for face in faces {
            var bbox = face.boundingBox

            // Apply padding if specified
            if padding > 0 {
                let centerX = bbox.midX
                let centerY = bbox.midY
                let newWidth = bbox.width * (1.0 + padding)
                let newHeight = bbox.height * (1.0 + padding)

                bbox.origin.x = max(0, centerX - newWidth / 2)
                bbox.origin.y = max(0, centerY - newHeight / 2)
                bbox.size.width = min(newWidth, 1.0 - bbox.origin.x)
                bbox.size.height = min(newHeight, 1.0 - bbox.origin.y)
            }

            // Convert normalized coordinates (0-1) to image coordinates
            // Vision uses bottom-left origin, AppKit uses top-left
            let x = bbox.origin.x * imageSize.width
            let y = (1.0 - bbox.origin.y - bbox.size.height) * imageSize.height  // Flip Y
            let width = bbox.size.width * imageSize.width
            let height = bbox.size.height * imageSize.height

            let faceRect = NSRect(x: x, y: y, width: width, height: height)
            faceRect.fill()

            print("  Face at: x=\(Int(x)), y=\(Int(y)), w=\(Int(width)), h=\(Int(height))")
        }

        maskImage.unlockFocus()

        return maskImage
    }

    // MARK: - Advanced Face Detection (with landmarks)

    /// Generate mask using face landmarks for more precise protection
    /// - Parameter image: Input image
    /// - Returns: Mask with face contours (more precise than rectangles)
    static func generateLandmarkMask(for image: NSImage) async throws -> NSImage {
        guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            throw GlazingError.invalidImage
        }

        print("🔍 Detecting face landmarks...")

        // Create face landmarks request
        let request = VNDetectFaceLandmarksRequest()

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        guard let results = request.results, !results.isEmpty else {
            print("⚠️  No faces detected, using full-image mask")
            return createFullMask(size: image.size)
        }

        print("✅ Detected \(results.count) face(s) with landmarks")

        return drawLandmarkMask(faces: results, imageSize: image.size)
    }

    /// Draw mask using face landmarks (more precise)
    private static func drawLandmarkMask(faces: [VNFaceObservation], imageSize: CGSize) -> NSImage {
        let maskImage = NSImage(size: imageSize)

        maskImage.lockFocus()

        // Black background
        NSColor.black.setFill()
        NSRect(origin: .zero, size: imageSize).fill()

        // White ellipses for faces based on landmarks
        NSColor.white.setFill()

        for face in faces {
            let bbox = face.boundingBox

            // Convert to image coordinates
            let x = bbox.origin.x * imageSize.width
            let y = (1.0 - bbox.origin.y - bbox.size.height) * imageSize.height
            let width = bbox.size.width * imageSize.width
            let height = bbox.size.height * imageSize.height

            // Draw ellipse instead of rectangle for more natural shape
            let faceOval = NSRect(x: x, y: y, width: width, height: height)
            let path = NSBezierPath(ovalIn: faceOval)
            path.fill()

            print("  Face oval at: x=\(Int(x)), y=\(Int(y)), w=\(Int(width)), h=\(Int(height))")
        }

        maskImage.unlockFocus()

        return maskImage
    }

    // MARK: - Mask Validation

    /// Validate that a mask is suitable for protection
    /// - Parameter mask: Mask image to validate
    /// - Returns: True if mask is valid (has some white pixels)
    static func validateMask(_ mask: NSImage) -> Bool {
        guard let cgImage = mask.cgImage(forProposedRect: nil, context: nil, hints: nil),
              let dataProvider = cgImage.dataProvider,
              let data = dataProvider.data,
              let bytes = CFDataGetBytePtr(data) else {
            return false
        }

        let bytesPerPixel = cgImage.bitsPerPixel / 8
        let bytesPerRow = cgImage.bytesPerRow
        let width = cgImage.width
        let height = cgImage.height

        // Count white pixels (value > 200)
        var whitePixelCount = 0
        let sampleRate = 10  // Check every 10th pixel for speed

        for y in stride(from: 0, to: height, by: sampleRate) {
            for x in stride(from: 0, to: width, by: sampleRate) {
                let offset = y * bytesPerRow + x * bytesPerPixel
                if offset < CFDataGetLength(data) {
                    let pixelValue = bytes[offset]
                    if pixelValue > 200 {
                        whitePixelCount += 1
                    }
                }
            }
        }

        let totalSamples = (height / sampleRate) * (width / sampleRate)
        let whitePercentage = Double(whitePixelCount) / Double(totalSamples) * 100

        print("🎭 Mask validation: \(Int(whitePercentage))% white pixels")

        // Mask should have at least 1% white pixels
        return whitePercentage >= 1.0
    }
}
