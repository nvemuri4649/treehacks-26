//
// MarkdownText.swift
// Cena
//
// Renders markdown content with proper formatting:
// - Inline: **bold**, *italic*, `code`, [links](url)
// - Code blocks (``` fenced)
// - Bullet / numbered lists
// - Headings
//
// Keeps the translucent glass aesthetic.
//

import SwiftUI

struct MarkdownText: View {
    let text: String
    var fontSize: CGFloat = 13

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                switch block {
                case .paragraph(let content):
                    inlineMarkdown(content)

                case .codeBlock(let lang, let code):
                    codeBlockView(language: lang, code: code)

                case .heading(let level, let content):
                    headingView(level: level, content: content)
                }
            }
        }
    }

    //MARK: - Block parsing

    private enum Block {
        case paragraph(String)
        case codeBlock(language: String?, code: String)
        case heading(level: Int, content: String)
    }

    private var blocks: [Block] {
        var result: [Block] = []
        let lines = text.components(separatedBy: "\n")
        var i = 0

        while i < lines.count {
            let line = lines[i]

            //Fenced code block
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                let lang = line.trimmingCharacters(in: .whitespaces)
                    .dropFirst(3)
                    .trimmingCharacters(in: .whitespaces)
                let language = lang.isEmpty ? nil : lang
                var codeLines: [String] = []
                i += 1
                while i < lines.count {
                    let cl = lines[i]
                    if cl.trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                        i += 1
                        break
                    }
                    codeLines.append(cl)
                    i += 1
                }
                result.append(.codeBlock(language: language, code: codeLines.joined(separator: "\n")))
                continue
            }

            //Heading (# ... ####)
            if let match = line.range(of: #"^(#{1,4})\s+(.+)$"#, options: .regularExpression) {
                let raw = String(line[match])
                let hashes = raw.prefix(while: { $0 == "#" }).count
                let content = raw.drop(while: { $0 == "#" }).trimmingCharacters(in: .whitespaces)
                result.append(.heading(level: hashes, content: content))
                i += 1
                continue
            }

            //Accumulate paragraph lines (including blank lines as breaks)
            var paraLines: [String] = []
            while i < lines.count {
                let pl = lines[i]
                //Stop at code fence or heading
                if pl.trimmingCharacters(in: .whitespaces).hasPrefix("```") { break }
                if pl.range(of: #"^#{1,4}\s+"#, options: .regularExpression) != nil { break }
                paraLines.append(pl)
                i += 1
            }

            let joined = paraLines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            if !joined.isEmpty {
                result.append(.paragraph(joined))
            }
        }

        return result
    }

    //MARK: - Inline markdown

    @ViewBuilder
    private func inlineMarkdown(_ content: String) -> some View {
        if let attributed = try? AttributedString(
            markdown: content,
            options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        ) {
            Text(attributed)
                .font(.system(size: fontSize))
                .tint(.blue.opacity(0.8))
        } else {
            Text(content)
                .font(.system(size: fontSize))
        }
    }

    //MARK: - Code block

    private func codeBlockView(language: String?, code: String) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            if let lang = language, !lang.isEmpty {
                HStack {
                    Text(lang)
                        .font(.system(size: 9, weight: .medium, design: .monospaced))
                        .foregroundStyle(.secondary.opacity(0.6))
                    Spacer()
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(code, forType: .string)
                    } label: {
                        Image(systemName: "doc.on.doc")
                            .font(.system(size: 9))
                            .foregroundStyle(.secondary.opacity(0.5))
                    }
                    .buttonStyle(.plain)
                    .help("Copy code")
                }
                .padding(.horizontal, 10)
                .padding(.top, 7)
                .padding(.bottom, 2)
            }

            ScrollView(.horizontal, showsIndicators: false) {
                Text(code)
                    .font(.system(size: 11.5, design: .monospaced))
                    .foregroundStyle(.primary.opacity(0.85))
                    .padding(.horizontal, 10)
                    .padding(.vertical, language != nil ? 6 : 10)
                    .textSelection(.enabled)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.black.opacity(0.25), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .stroke(.white.opacity(0.06), lineWidth: 0.5)
        )
    }

    //MARK: - Heading

    @ViewBuilder
    private func headingView(level: Int, content: String) -> some View {
        let size: CGFloat = level == 1 ? fontSize + 5 :
                            level == 2 ? fontSize + 3 :
                            level == 3 ? fontSize + 1 : fontSize
        let weight: Font.Weight = level <= 2 ? .bold : .semibold

        if let attributed = try? AttributedString(
            markdown: content,
            options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        ) {
            Text(attributed)
                .font(.system(size: size, weight: weight))
        } else {
            Text(content)
                .font(.system(size: size, weight: weight))
        }
    }
}
