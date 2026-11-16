import Foundation
import AppKit
import SwiftUI

enum PythonTokenKind: Equatable { case keyword, string, comment, number, name }
struct PythonToken: Equatable { let range:NSRange;let kind:PythonTokenKind }
enum PythonSyntax {
    private static let keywords=Set("False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield".split(separator:" ").map(String.init))
    private static let expression=try! NSRegularExpression(pattern: #"(?s:\"\"\".*?\"\"\"|'''.*?''')|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|#[^\n]*|\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b|\b[A-Za-z_][A-Za-z_0-9]*\b"#)
    static func tokens(in source:String)->[PythonToken] {
        let text=source as NSString
        return expression.matches(in:source,range:NSRange(location:0,length:text.length)).compactMap { match in
            let value=text.substring(with:match.range)
            let kind:PythonTokenKind
            if value.hasPrefix("#") {kind = .comment}
            else if value.hasPrefix("\"") || value.hasPrefix("'") {kind = .string}
            else if value.first?.isNumber == true {kind = .number}
            else if keywords.contains(value) {kind = .keyword}
            else {
                let suffix=text.substring(from:NSMaxRange(match.range)).drop(while:{$0 == " " || $0 == "\t"})
                guard suffix.first == "(" else {return nil}
                kind = .name
            }
            return PythonToken(range:match.range,kind:kind)
        }
    }
    static func attributed(_ source:String,palette supplied:AppearanceNativePalette?=nil)->NSAttributedString {
        let palette=supplied ?? AppearanceThemePolicy(preferences:.factory,colorScheme:.dark,increasedContrast:false,reduceTransparency:false).native
        let result=NSMutableAttributedString(string:source,attributes:[.font:NSFont.monospacedSystemFont(ofSize:12,weight:.regular),.foregroundColor:palette.text])
        for token in tokens(in:source) {
            let color:NSColor
            switch token.kind {
            case .keyword:color=palette.keyword
            case .string:color=palette.string
            case .comment:color=palette.comment
            case .number:color=palette.number
            case .name:color=palette.name
            }
            result.addAttribute(.foregroundColor,value:color,range:token.range)
        }
        return result
    }
}

/// Native read-only text selection and scrolling. Display never modifies exported bytes.
struct SyntaxHighlightedCode:NSViewRepresentable {
    let source:String
    @Environment(\.appearanceTheme) private var theme
    func makeCoordinator()->Coordinator {Coordinator()}
    @MainActor final class Coordinator {
        private var lastSource:String?
        private var lastPaletteFingerprint:String?
        func update(_ scroll:NSScrollView,source:String,theme:AppearanceThemePolicy) {
            guard let text=scroll.documentView as? NSTextView else{return}
            let sourceChanged=lastSource != source
            let paletteChanged=lastPaletteFingerprint != theme.syntaxPaletteFingerprint
            let selection=text.selectedRanges,origin=scroll.contentView.bounds.origin
            let palette=theme.native
            scroll.backgroundColor=palette.background
            text.backgroundColor=palette.background
            text.insertionPointColor=palette.text
            if sourceChanged || paletteChanged {
                text.textStorage?.setAttributedString(PythonSyntax.attributed(source,palette:palette))
                lastSource=source;lastPaletteFingerprint=theme.syntaxPaletteFingerprint
            }
            if let container=text.textContainer,let manager=text.layoutManager {
                manager.ensureLayout(for:container)
                let used=manager.usedRect(for:container)
                text.setFrameSize(NSSize(width:max(scroll.contentSize.width,used.width+32),height:max(scroll.contentSize.height,used.height+32)))
            }
            if sourceChanged {
                text.scrollRangeToVisible(NSRange(location:0,length:0))
            } else if paletteChanged {
                text.selectedRanges=selection
                scroll.contentView.scroll(to:origin)
                scroll.reflectScrolledClipView(scroll.contentView)
            }
            NativeScrollChrome.hideSystemScrollers(in:scroll)
        }
    }
    func makeNSView(context:Context)->NSScrollView {
        Self.makeScrollView(theme:theme)
    }
    @MainActor static func makeScrollView(theme:AppearanceThemePolicy)->NSScrollView {
        let scroll=NSScrollView();scroll.borderType = .noBorder
        scroll.drawsBackground=true;scroll.backgroundColor=theme.native.background
        NativeScrollChrome.hideSystemScrollers(in:scroll)
        let text=NSTextView(frame:.zero)
        text.isEditable=false;text.isSelectable=true;text.isRichText=false
        text.drawsBackground=true;text.backgroundColor=scroll.backgroundColor
        text.insertionPointColor = theme.native.text;text.textContainerInset=NSSize(width:16,height:16)
        text.isVerticallyResizable=true;text.isHorizontallyResizable=true
        text.minSize = .zero;text.maxSize=NSSize(width:CGFloat.greatestFiniteMagnitude,height:CGFloat.greatestFiniteMagnitude)
        text.textContainer?.widthTracksTextView=false
        text.textContainer?.containerSize=NSSize(width:CGFloat.greatestFiniteMagnitude,height:CGFloat.greatestFiniteMagnitude)
        text.setAccessibilityLabel("Read-only source code")
        scroll.documentView=text
        return scroll
    }
    func updateNSView(_ scroll:NSScrollView,context:Context) {
        context.coordinator.update(scroll,source:source,theme:theme)
    }
}
