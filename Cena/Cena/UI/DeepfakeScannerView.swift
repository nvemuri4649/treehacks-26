//
//  DeepfakeScannerView.swift
//  Cena
//

import SwiftUI
import WebKit

struct DeepfakeScannerView: View {
    @State private var isLoading = true
    @State private var loadError: String?

    private let url = URL(string: "http://127.0.0.1:8000/deepfake/")!

    var body: some View {
        ZStack {
            VisualEffectViewRepresentable(material: .hudWindow, blendingMode: .behindWindow)
                .ignoresSafeArea()

            WebViewRepresentable(url: url, isLoading: $isLoading, loadError: $loadError)
                .opacity(isLoading ? 0.3 : 1.0)

            if isLoading && loadError == nil {
                VStack(spacing: 12) {
                    CenaLogo(size: 32, isAnimating: true, color: .white.opacity(0.6))
                    Text("Loading scanner…")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }
            }

            if let error = loadError {
                VStack(spacing: 10) {
                    CenaLogo(size: 36, isAnimating: false, color: .white.opacity(0.3))
                    Text("Scanner Unavailable")
                        .font(.system(size: 14, weight: .medium))
                    Text(error)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 280)
                }
            }
        }
        .frame(minWidth: 700, minHeight: 600)
    }
}

// MARK: - WKWebView wrapper with file upload support

struct WebViewRepresentable: NSViewRepresentable {
    let url: URL
    @Binding var isLoading: Bool
    @Binding var loadError: String?

    func makeNSView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .default()

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = context.coordinator
        webView.uiDelegate = context.coordinator
        webView.setValue(false, forKey: "drawsBackground")
        webView.load(URLRequest(url: url))
        return webView
    }

    func updateNSView(_ nsView: WKWebView, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    class Coordinator: NSObject, WKNavigationDelegate, WKUIDelegate {
        let parent: WebViewRepresentable
        init(_ parent: WebViewRepresentable) { self.parent = parent }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            Task { @MainActor in
                parent.isLoading = false
                parent.loadError = nil
            }
        }

        func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
            Task { @MainActor in
                parent.isLoading = false
                parent.loadError = error.localizedDescription
            }
        }

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
            Task { @MainActor in
                parent.isLoading = false
                parent.loadError = "Could not connect to server.\nMake sure python -m server.main is running."
            }
        }

        // File upload support — handles <input type="file">
        func webView(
            _ webView: WKWebView,
            runOpenPanelWith parameters: WKOpenPanelParameters,
            initiatedByFrame frame: WKFrameInfo,
            completionHandler: @escaping ([URL]?) -> Void
        ) {
            let panel = NSOpenPanel()
            panel.allowsMultipleSelection = parameters.allowsMultipleSelection
            panel.canChooseDirectories = parameters.allowsDirectories
            panel.canChooseFiles = true
            panel.begin { response in
                if response == .OK {
                    completionHandler(panel.urls)
                } else {
                    completionHandler(nil)
                }
            }
        }
    }
}
