//
// PasteboardMonitor.swift
// Cena
//
// Monitors the system pasteboard for copied images and triggers protection
//

import Foundation
import AppKit
import Combine

class PasteboardMonitor: ObservableObject {
    @Published var isMonitoring = false

    private var timer: Timer?
    private var lastChangeCount: Int
    private var onImageDetected: ((NSImage) -> Void)?

    init() {
        self.lastChangeCount = NSPasteboard.general.changeCount
    }

    //MARK: - Monitoring Control

    /// Start monitoring the pasteboard for image changes
    /// - Parameter onImageDetected: Callback when new image is detected
    func startMonitoring(onImageDetected: @escaping (NSImage) -> Void) {
        guard !isMonitoring else {
            print("⚠️  Pasteboard monitoring already active")
            return
        }

        self.onImageDetected = onImageDetected
        self.lastChangeCount = NSPasteboard.general.changeCount
        self.isMonitoring = true

        //Poll pasteboard every 0.5 seconds on the main RunLoop
        let t = Timer(timeInterval: 0.5, repeats: true) { [weak self] _ in
            self?.checkPasteboard()
        }
        RunLoop.main.add(t, forMode: .common)
        timer = t

        print("👀 Started pasteboard monitoring")
    }

    /// Stop monitoring the pasteboard
    func stopMonitoring() {
        timer?.invalidate()
        timer = nil
        isMonitoring = false
        print("🛑 Stopped pasteboard monitoring")
    }

    //MARK: - Pasteboard Checking

    /// Check if pasteboard has changed and contains an image
    private func checkPasteboard() {
        let pasteboard = NSPasteboard.general
        let currentCount = pasteboard.changeCount

        //No change detected
        guard currentCount != lastChangeCount else {
            return
        }

        print("📋 Pasteboard changed (\(lastChangeCount) → \(currentCount)), types: \(pasteboard.types?.map(\.rawValue) ?? [])")
        lastChangeCount = currentCount

        //Check for image content
        if let image = getImageFromPasteboard() {
            print("📋 Image detected on pasteboard: \(image.size)")
            DispatchQueue.main.async { [weak self] in
                self?.onImageDetected?(image)
            }
        } else {
            print("📋 No image found in pasteboard")
        }
    }

    /// Extract image from pasteboard if present
    /// - Returns: NSImage if pasteboard contains image data, nil otherwise
    private func getImageFromPasteboard() -> NSImage? {
        let pasteboard = NSPasteboard.general

        //Try multiple formats in order of preference

        //1. NSImage objects (highest quality)
        if let images = pasteboard.readObjects(forClasses: [NSImage.self], options: nil) as? [NSImage],
           let image = images.first {
            print("  Found NSImage object")
            return image
        }

        //2. PNG data
        if let pngData = pasteboard.data(forType: .png),
           let image = NSImage(data: pngData) {
            print("  Found PNG data")
            return image
        }

        //3. TIFF data
        if let tiffData = pasteboard.data(forType: .tiff),
           let image = NSImage(data: tiffData) {
            print("  Found TIFF data")
            return image
        }

        //4. JPEG data (less common for clipboard)
        if let jpegData = pasteboard.data(forType: NSPasteboard.PasteboardType("public.jpeg")),
           let image = NSImage(data: jpegData) {
            print("  Found JPEG data")
            return image
        }

        //5. File URLs (dragged files)
        if let urls = pasteboard.readObjects(forClasses: [NSURL.self], options: nil) as? [URL] {
            for url in urls {
                if let image = loadImageFromURL(url) {
                    print("  Found image file: \(url.lastPathComponent)")
                    return image
                }
            }
        }

        return nil
    }

    /// Load image from file URL
    private func loadImageFromURL(_ url: URL) -> NSImage? {
        //Check if file is an image
        let imageExtensions = ["png", "jpg", "jpeg", "tiff", "tif", "gif", "bmp", "heic", "heif"]
        let ext = url.pathExtension.lowercased()

        guard imageExtensions.contains(ext) else {
            return nil
        }

        return NSImage(contentsOf: url)
    }

    //MARK: - Pasteboard Manipulation

    /// Replace the current pasteboard image with a protected version
    /// - Parameter protectedImage: encrypted image to put on pasteboard
    func replacePasteboardImage(with protectedImage: NSImage) {
        let pasteboard = NSPasteboard.general

        pasteboard.clearContents()

        //Add PNG representation (best quality)
        if let pngData = protectedImage.pngData() {
            pasteboard.setData(pngData, forType: .png)
            print("✅ Replaced pasteboard with protected image")

            //Update our change count to avoid re-detecting our own change
            lastChangeCount = pasteboard.changeCount
        } else {
            print("❌ Failed to encode protected image for pasteboard")
        }
    }

    //MARK: - Utilities

    /// Check if pasteboard currently contains an image
    var containsImage: Bool {
        return getImageFromPasteboard() != nil
    }

    /// Get current pasteboard image without triggering callbacks
    func getCurrentImage() -> NSImage? {
        return getImageFromPasteboard()
    }
}

//MARK: - Pasteboard Type Extensions

extension NSPasteboard.PasteboardType {
    static let jpg = NSPasteboard.PasteboardType("public.jpeg")
    static let png = NSPasteboard.PasteboardType("public.png")
}
