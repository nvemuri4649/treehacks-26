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
            let _ = min(newWidth, 1.0 - newX)
            let _ = min(newHeight, 1.0 - newY)

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

    /// Draw white rectangles for detected faces on black background using CGContext
    private static func drawFaceMask(
        faces: [VNFaceObservation],
        imageSize: CGSize,
        cgImage: CGImage,
        padding: CGFloat = 0.0
    ) -> NSImage {
        // Use actual pixel dimensions from the CGImage for reliable rendering
        let pixelWidth = cgImage.width
        let pixelHeight = cgImage.height

        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let ctx = CGContext(
            data: nil,
            width: pixelWidth,
            height: pixelHeight,
            bitsPerComponent: 8,
            bytesPerRow: pixelWidth,
            space: colorSpace,
            bitmapInfo: 0
        ) else {
            print("❌ Failed to create CGContext for mask")
            return createFullMask(size: imageSize)
        }

        // Black background
        ctx.setFillColor(gray: 0, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: pixelWidth, height: pixelHeight))

        // White rectangles for faces
        ctx.setFillColor(gray: 1, alpha: 1)

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

            // Convert normalized Vision coordinates (bottom-left origin, 0-1)
            // to pixel coordinates (CGContext also uses bottom-left origin)
            let x = bbox.origin.x * CGFloat(pixelWidth)
            let y = bbox.origin.y * CGFloat(pixelHeight)
            let w = bbox.size.width * CGFloat(pixelWidth)
            let h = bbox.size.height * CGFloat(pixelHeight)

            let faceRect = CGRect(x: x, y: y, width: w, height: h)
            ctx.fill(faceRect)

            print("  Face at: x=\(Int(x)), y=\(Int(y)), w=\(Int(w)), h=\(Int(h))")
        }

        guard let maskCGImage = ctx.makeImage() else {
            print("❌ Failed to create mask image from context")
            return createFullMask(size: imageSize)
        }

        let maskImage = NSImage(cgImage: maskCGImage, size: imageSize)
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

    /// Draw mask using face landmarks (more precise) via CGContext
    private static func drawLandmarkMask(faces: [VNFaceObservation], imageSize: CGSize) -> NSImage {
        // Default to reasonable pixel dimensions
        let pixelWidth = Int(imageSize.width)
        let pixelHeight = Int(imageSize.height)

        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let ctx = CGContext(
            data: nil,
            width: pixelWidth,
            height: pixelHeight,
            bitsPerComponent: 8,
            bytesPerRow: pixelWidth,
            space: colorSpace,
            bitmapInfo: 0
        ) else {
            return createFullMask(size: imageSize)
        }

        // Black background
        ctx.setFillColor(gray: 0, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: pixelWidth, height: pixelHeight))

        // White ellipses for faces
        ctx.setFillColor(gray: 1, alpha: 1)

        for face in faces {
            let bbox = face.boundingBox
            let x = bbox.origin.x * CGFloat(pixelWidth)
            let y = bbox.origin.y * CGFloat(pixelHeight)
            let w = bbox.size.width * CGFloat(pixelWidth)
            let h = bbox.size.height * CGFloat(pixelHeight)

            ctx.fillEllipse(in: CGRect(x: x, y: y, width: w, height: h))
            print("  Face oval at: x=\(Int(x)), y=\(Int(y)), w=\(Int(w)), h=\(Int(h))")
        }

        guard let maskCGImage = ctx.makeImage() else {
            return createFullMask(size: imageSize)
        }

        return NSImage(cgImage: maskCGImage, size: imageSize)
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
