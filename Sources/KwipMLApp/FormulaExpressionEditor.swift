import AppKit
import SwiftUI

@MainActor final class FormulaEditorCommands:ObservableObject {
    weak var textView:NSTextView?
    func insert(_ text:String) {guard let textView else {return};textView.insertText(text,replacementRange:textView.selectedRange());textView.window?.makeFirstResponder(textView)}
}
struct FormulaExpressionEditor:NSViewRepresentable {
    @Binding var text:String
    let commands:FormulaEditorCommands
    func makeCoordinator()->Coordinator {Coordinator(self)}
    func makeNSView(context:Context)->NSScrollView {
        let scroll=NSScrollView();scroll.hasVerticalScroller=true;scroll.borderType = .noBorder
        let view=NSTextView();view.isRichText=false;view.isAutomaticQuoteSubstitutionEnabled=false;view.isAutomaticDashSubstitutionEnabled=false;view.isAutomaticTextReplacementEnabled=false;view.font = .monospacedSystemFont(ofSize:12,weight:.regular);view.textContainerInset=NSSize(width:5,height:7);view.backgroundColor = .textBackgroundColor;view.textColor = .textColor;view.isVerticallyResizable=true;view.autoresizingMask=[.width];view.textContainer?.widthTracksTextView=true;view.delegate=context.coordinator;view.string=text;view.setAccessibilityLabel("Formula expression");scroll.documentView=view;commands.textView=view;return scroll
    }
    func updateNSView(_ view:NSScrollView,context:Context) {context.coordinator.parent=self;if let textView=view.documentView as? NSTextView {commands.textView=textView;if textView.string != text {textView.string=text}}}
    final class Coordinator:NSObject,NSTextViewDelegate {
        var parent:FormulaExpressionEditor
        init(_ parent:FormulaExpressionEditor) {self.parent=parent}
        func textDidChange(_ notification:Notification) {if let view=notification.object as? NSTextView {parent.text=view.string}}
    }
}
