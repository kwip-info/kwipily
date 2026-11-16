import SwiftUI
import Foundation

/// Presentation-only blocks. Scientific content and assessment visibility remain
/// owned by each lesson; this renderer never creates hints or correctness claims.
enum LessonTextLayout {
    enum Kind: Equatable { case heading, paragraph, bullet, code, numbered(Int) }
    struct Block: Identifiable {
        let kind: Kind
        let text: String
        let id: Int
    }
    private static let sentences=try! NSRegularExpression(pattern:"(?<=[.!?])\\s+(?=[A-Z])")
    static func blocks(_ text: String) -> [Block] {
        var output: [Block] = []
        var prose: [String] = []
        var code: [String] = []
        var inCode = false
        func append(_ kind: Kind, _ value: String) {
            guard !value.isEmpty else { return }
            output.append(Block(kind: kind, text: value, id: output.count))
        }
        func flush() {
            let value = prose.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            prose.removeAll()
            guard !value.isEmpty else { return }
            // Legacy public references are plain prose. Break long runs at sentence
            // boundaries without splitting decimal numbers or rewriting formulas.
            let ranges = sentences.matches(in:value,range:NSRange(value.startIndex...,in:value))
            var start = value.startIndex
            var paragraph = ""
            for match in ranges {
                guard let range = Range(match.range, in: value) else { continue }
                let sentence = String(value[start..<range.lowerBound])
                paragraph += (paragraph.isEmpty ? "" : " ") + sentence
                if paragraph.count >= 170 { append(.paragraph, paragraph); paragraph = "" }
                start = range.upperBound
            }
            paragraph += (paragraph.isEmpty ? "" : " ") + String(value[start...])
            append(.paragraph, paragraph)
        }
        for line in text.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("```") {
                flush()
                if inCode { append(.code, code.joined(separator: "\n")); code.removeAll() }
                inCode.toggle(); continue
            }
            if inCode { code.append(line); continue }
            if trimmed.isEmpty { flush(); continue }
            if trimmed.hasPrefix("#") {
                flush(); append(.heading, String(trimmed.drop(while: { $0 == "#" || $0 == " " }))); continue
            }
            if trimmed.hasPrefix("- ") || trimmed.hasPrefix("* ") {
                flush(); append(.bullet, String(trimmed.dropFirst(2))); continue
            }
            if let match=trimmed.range(of:#"^[0-9]{1,3}[.)]\s+"#,options:.regularExpression),
               let number=Int(trimmed[..<match.upperBound].prefix(while:{$0.isNumber})) {
                flush();append(.numbered(number),String(trimmed[match.upperBound...]));continue
            }
            prose.append(line)
        }
        flush()
        if inCode { append(.code, code.joined(separator: "\n")) }
        return output
    }
}

enum LessonRichTextPresentation {
    struct Block:Identifiable {let source:LessonTextLayout.Block;let inline:AttributedString?;var id:Int {source.id};var kind:LessonTextLayout.Kind {source.kind};var text:String {source.text}}
    private static let cache=BoundedPresentationCache<String,[Block]>(maximumEntries:128,maximumCost:2_000_000)
    static func blocks(_ text:String)->[Block] {
        cache.value(key:text,input:text,cost:max(1,text.utf8.count*4)) {
            LessonTextLayout.blocks(text).map {block in
                let value=block.text
                let inline:AttributedString?
                switch block.kind {
                case .heading,.code:inline=nil
                default:inline=(value.contains("**") || value.contains("`") || value.contains("](")) ? try? AttributedString(markdown:value,options:.init(interpretedSyntax:.inlineOnlyPreservingWhitespace)):nil
                }
                return Block(source:block,inline:inline)
            }
        }
    }
}
struct LessonRichText: View {
    @Environment(\.lessonTypography) private var typography
    let text: String
    @Environment(\.lessonContentAccent) private var lessonAccent
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(LessonRichTextPresentation.blocks(text)) { block in
                switch block.kind {
                case .heading:
                    Text(block.text).font(.system(size:typography.size(13), weight: .semibold))
                case .paragraph:
                    formatted(block).font(.system(size:typography.size(12))).lineSpacing(3)
                case .bullet:
                    HStack(alignment: .top, spacing: 9) {
                        Rectangle().fill(lessonAccent).frame(width: 4, height: 4).padding(.top, 7)
                        formatted(block).font(.system(size:typography.size(12))).lineSpacing(3)
                    }
                case .numbered(let number):
                    HStack(alignment:.top,spacing:9) {
                        Text(String(number)).font(.system(size:typography.size(10),weight:.semibold)).foregroundStyle(lessonAccent)
                            .frame(minWidth:20,minHeight:20).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule)).padding(.top,1)
                        formatted(block).font(.system(size:typography.size(12))).lineSpacing(3).fixedSize(horizontal:false,vertical:true)
                    }
                case .code:
                    Text(block.text).font(.system(size:typography.size(12), design: .monospaced)).lineSpacing(5)
                        .frame(maxWidth: .infinity, alignment: .leading).padding(12)
                        .background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule))
                }
            }
        }.frame(maxWidth: 840, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .leading).fixedSize(horizontal:false,vertical:true).textSelection(.enabled)
    }
    private func formatted(_ block:LessonRichTextPresentation.Block)->Text {
        if let inline=block.inline {return Text(inline)}
        return Text(block.text)
    }
}
struct LessonTextPanel: View {
    let title: String
    let text: String
    var body: some View {
        LessonBlockSection(title:title,symbol:"book.closed",identifier:"lesson.text."+title) {LessonRichText(text:text)}
    }
}
