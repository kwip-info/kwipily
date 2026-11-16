import AppKit
import SwiftUI

/// Presentation only. Lexical colors are not a parser or evidence that a rule is valid.
enum GameRuleCodeLanguage { case json, expression }
enum GameRuleSyntax {
    private static let json = try! NSRegularExpression(pattern: #"\"(?:\\.|[^\"\\])*\"|-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b|\b(?:true|false|null)\b"#)
    private static let references=try! NSRegularExpression(pattern:#"\b(?:state|action|params|next|outcome|actor|player)\b"#)
    static func attributed(_ source:String,language:GameRuleCodeLanguage,palette:AppearanceNativePalette)->NSAttributedString {
        if language == .expression {
            let result=NSMutableAttributedString(attributedString:PythonSyntax.attributed(source,palette:palette))
            let excluded=PythonSyntax.tokens(in:source).filter{$0.kind == .string || $0.kind == .comment}.map(\.range)
            for match in references.matches(in:source,range:NSRange(location:0,length:(source as NSString).length)) where !excluded.contains(where:{NSIntersectionRange($0,match.range).length>0}) {
                result.addAttribute(.foregroundColor,value:palette.name,range:match.range)
            }
            return result
        }
        let result=NSMutableAttributedString(string:source,attributes:[.font:NSFont.monospacedSystemFont(ofSize:12,weight:.regular),.foregroundColor:palette.text])
        let text=source as NSString
        for token in json.matches(in:source,range:NSRange(location:0,length:text.length)) {
            let value=text.substring(with:token.range)
            let color:NSColor
            if value.hasPrefix("\"") {
                var end=NSMaxRange(token.range)
                while end<text.length,let scalar=UnicodeScalar(text.character(at:end)),CharacterSet.whitespacesAndNewlines.contains(scalar){end+=1}
                color=end<text.length && text.character(at:end)==58 ? palette.name:palette.string
            } else {color=["true","false","null"].contains(value) ? palette.keyword:palette.number}
            result.addAttribute(.foregroundColor,value:color,range:token.range)
        }
        return result
    }
}

/// Each buffer owns its text undo stack. Revert/import must never clear another
/// field's or the document window's undo manager.
@MainActor private final class GameRuleNativeTextView:NSTextView {
    private let bufferUndo:UndoManager = {let manager=UndoManager();manager.levelsOfUndo=100;return manager}()
    override var undoManager:UndoManager? {bufferUndo}
}

/// Editable, wrapped native text with bounded debounced highlighting. Attribute updates
/// preserve selection, viewport, exact bytes and the native editor's undo operations.
struct GameRuleCodeEditor:NSViewRepresentable {
    @Binding var text:String
    let language:GameRuleCodeLanguage
    let label:String
    var limit:Int=140000
    @Environment(\.appearanceTheme) private var theme
    func makeCoordinator()->Coordinator {Coordinator(self)}
    func makeNSView(context:Context)->NSScrollView {
        let scroll=Self.makeScrollView(label:label)
        if let view=scroll.documentView as? NSTextView {context.coordinator.attach(view)}
        return scroll
    }
    func updateNSView(_ scroll:NSScrollView,context:Context) {
        context.coordinator.parent=self
        context.coordinator.update(scroll,theme:theme)
    }
    static func dismantleNSView(_ scroll:NSScrollView,coordinator:Coordinator) {coordinator.detach();(scroll.documentView as? NSTextView)?.delegate=nil}
    @MainActor static func makeScrollView(label:String)->NSScrollView {
        let scroll=NSScrollView();scroll.borderType = .noBorder;scroll.hasVerticalScroller=true;scroll.autohidesScrollers=true
        let view=GameRuleNativeTextView(frame:NSRect(x:0,y:0,width:400,height:80))
        view.isRichText=false;view.allowsUndo=true;view.isAutomaticQuoteSubstitutionEnabled=false
        view.isAutomaticDashSubstitutionEnabled=false;view.isAutomaticTextReplacementEnabled=false
        view.isAutomaticSpellingCorrectionEnabled=false;view.isContinuousSpellCheckingEnabled=false
        view.font = .monospacedSystemFont(ofSize:12,weight:.regular)
        view.textContainerInset=NSSize(width:12,height:12)
        view.isVerticallyResizable=true;view.isHorizontallyResizable=false
        view.minSize = .zero;view.maxSize=NSSize(width:CGFloat.greatestFiniteMagnitude,height:CGFloat.greatestFiniteMagnitude)
        view.autoresizingMask=[.width];view.textContainer?.widthTracksTextView=true
        view.textContainer?.containerSize=NSSize(width:400,height:CGFloat.greatestFiniteMagnitude)
        view.setAccessibilityLabel(label);scroll.documentView=view
        return scroll
    }
    @MainActor final class Coordinator:NSObject,NSTextViewDelegate {
        var parent:GameRuleCodeEditor
        var pending:DispatchWorkItem?
        private var palette:AppearanceNativePalette?
        private var fingerprint:String?
        private var changing=false
        private var undoObservations:[NSObjectProtocol]=[]
        init(_ parent:GameRuleCodeEditor){self.parent=parent}
        func attach(_ view:NSTextView) {
            detach();view.delegate=self
            undoObservations=[Notification.Name.NSUndoManagerDidUndoChange,.NSUndoManagerDidRedoChange].map {name in
                NotificationCenter.default.addObserver(forName:name,object:view.undoManager,queue:.main) {[weak self,weak view] _ in
                    MainActor.assumeIsolated {guard let self,let view else{return};self.textDidChange(Notification(name:NSText.didChangeNotification,object:view))}
                }
            }
        }
        func detach(){pending?.cancel();undoObservations.forEach{NotificationCenter.default.removeObserver($0)};undoObservations=[]}
        deinit {undoObservations.forEach{NotificationCenter.default.removeObserver($0)}}
        func update(_ scroll:NSScrollView,theme:AppearanceThemePolicy) {
            guard let view=scroll.documentView as? NSTextView else{return}
            view.setAccessibilityLabel(parent.label)
            palette=theme.native
            view.backgroundColor=theme.native.background;scroll.backgroundColor=theme.native.background
            view.insertionPointColor=theme.native.text
            let sourceChanged=view.string != parent.text
            if sourceChanged {
                // External changes (Revert/import/section switch) are a new editing buffer.
                changing=true;view.string=parent.text;view.setSelectedRange(NSRange(location:0,length:0));scroll.contentView.scroll(to:.zero);scroll.reflectScrolledClipView(scroll.contentView);view.undoManager?.removeAllActions();changing=false
            }
            if sourceChanged || fingerprint != theme.syntaxPaletteFingerprint {
                pending?.cancel();highlight(view);fingerprint=theme.syntaxPaletteFingerprint
            }
        }
        func textView(_ textView:NSTextView,shouldChangeTextIn affectedCharRange:NSRange,replacementString:String?)->Bool {
            guard let replacementString else{return true}
            return textView.string.utf16.count-affectedCharRange.length+replacementString.utf16.count <= parent.limit
        }
        func textDidChange(_ notification:Notification) {
            guard !changing,let view=notification.object as? NSTextView else{return}
            parent.text=view.string
            pending?.cancel()
            let work=DispatchWorkItem{[weak self,weak view] in guard let self,let view else{return};self.highlight(view)}
            pending=work;DispatchQueue.main.asyncAfter(deadline:.now()+0.12,execute:work)
        }
        private func highlight(_ view:NSTextView) {
            guard !view.hasMarkedText(),let palette,let storage=view.textStorage else{return}
            let selection=view.selectedRanges,origin=view.enclosingScrollView?.contentView.bounds.origin
            changing=true
            view.undoManager?.disableUndoRegistration()
            storage.beginEditing();storage.setAttributes([.font:NSFont.monospacedSystemFont(ofSize:12,weight:.regular),.foregroundColor:palette.text],range:NSRange(location:0,length:storage.length))
            let attributed=GameRuleSyntax.attributed(view.string,language:parent.language,palette:palette)
            attributed.enumerateAttribute(.foregroundColor,in:NSRange(location:0,length:attributed.length)){value,range,_ in if let value {storage.addAttribute(.foregroundColor,value:value,range:range)}}
            storage.endEditing();view.undoManager?.enableUndoRegistration()
            view.selectedRanges=selection
            if let origin,let scroll=view.enclosingScrollView {scroll.contentView.scroll(to:origin);scroll.reflectScrolledClipView(scroll.contentView)}
            view.typingAttributes=[.font:NSFont.monospacedSystemFont(ofSize:12,weight:.regular),.foregroundColor:palette.text]
            changing=false
        }
    }
}
