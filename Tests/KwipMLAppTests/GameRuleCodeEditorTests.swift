import XCTest
import AppKit
import SwiftUI
@testable import KwipMLApp

@MainActor final class GameRuleCodeEditorTests:XCTestCase {
    func testJSONHighlightingPreservesBytesAndDistinguishesKeysFromStrings() {
        let source = #"{"name":"🎲 rules","tests":[true,false,null,-1.25e2]}"#
        let theme = AppearanceThemePolicy(preferences:.factory,colorScheme:.dark,increasedContrast:false,reduceTransparency:false)
        let styled = GameRuleSyntax.attributed(source,language:.json,palette:theme.native)
        XCTAssertEqual(styled.string,source)
        let ns = source as NSString
        XCTAssertEqual(styled.attribute(.foregroundColor,at:ns.range(of:"\"name\"").location,effectiveRange:nil) as? NSColor,theme.native.name)
        XCTAssertEqual(styled.attribute(.foregroundColor,at:ns.range(of:"\"🎲 rules\"").location,effectiveRange:nil) as? NSColor,theme.native.string)
        XCTAssertEqual(styled.attribute(.foregroundColor,at:ns.range(of:"true").location,effectiveRange:nil) as? NSColor,theme.native.keyword)
        XCTAssertEqual(styled.attribute(.foregroundColor,at:ns.range(of:"-1.25e2").location,effectiveRange:nil) as? NSColor,theme.native.number)
    }
    func testExpressionsUseExistingThemeWithoutChangingScientificText() {
        let source = "1 if outcome.winner == player else -1\n# text remains exact"
        for scheme:ColorScheme in [.light,.dark] {
            let theme = AppearanceThemePolicy(preferences:.factory,colorScheme:scheme,increasedContrast:true,reduceTransparency:false)
            let styled = GameRuleSyntax.attributed(source,language:.expression,palette:theme.native)
            XCTAssertEqual(styled.string,source)
            XCTAssertEqual(styled.attribute(.foregroundColor,at:(source as NSString).range(of:"if").location,effectiveRange:nil) as? NSColor,theme.native.keyword)
        }
    }
    func testExpressionReferencesDoNotRecolorStringsOrComments() {
        let source="state.remaining + player if actor == 0 else len('state') # player"
        let theme=AppearanceThemePolicy(preferences:.factory,colorScheme:.dark,increasedContrast:false,reduceTransparency:false)
        let styled=GameRuleSyntax.attributed(source,language:.expression,palette:theme.native)
        XCTAssertEqual(styled.attribute(.foregroundColor,at:0,effectiveRange:nil) as? NSColor,theme.native.name)
        let ns=source as NSString
        XCTAssertEqual(styled.attribute(.foregroundColor,at:ns.range(of:"'state'").location+1,effectiveRange:nil) as? NSColor,theme.native.string)
        XCTAssertEqual(styled.attribute(.foregroundColor,at:ns.length-2,effectiveRange:nil) as? NSColor,theme.native.comment)
    }
    func testPresentationRefreshPreservesSelectionAndEnforcesEditBound() {
        var value = "state.remaining - action.take"
        let editor = GameRuleCodeEditor(text:Binding(get:{value},set:{value=$0}),language:.expression,label:"Test expression",limit:4096)
        let coordinator = GameRuleCodeEditor.Coordinator(editor)
        let scroll = GameRuleCodeEditor.makeScrollView(label:"Test expression")
        let text = scroll.documentView as! NSTextView
        text.string=value; text.setSelectedRange(NSRange(location:6,length:9))
        let selection=text.selectedRange()
        for scheme:ColorScheme in [.light,.dark] {
            let theme=AppearanceThemePolicy(preferences:.factory,colorScheme:scheme,increasedContrast:false,reduceTransparency:false)
            coordinator.update(scroll,theme:theme)
            XCTAssertEqual(text.string,value); XCTAssertEqual(text.selectedRange(),selection)
            coordinator.update(scroll,theme:theme)
            XCTAssertEqual(text.textStorage?.attribute(.foregroundColor,at:0,effectiveRange:nil) as? NSColor,theme.native.name)
        }
        XCTAssertFalse(coordinator.textView(text,shouldChangeTextIn:NSRange(location:0,length:0),replacementString:String(repeating:"x",count:4097)))
        XCTAssertTrue(coordinator.textView(text,shouldChangeTextIn:NSRange(location:0,length:5),replacementString:"next"))
    }
    func testEachBufferOwnsUndoAndHighlightingDoesNotUndoOrClearTextEdits() {
        var value="state.remaining"
        let editor=GameRuleCodeEditor(text:Binding(get:{value},set:{value=$0}),language:.expression,label:"Undo test",limit:4096)
        let coordinator=GameRuleCodeEditor.Coordinator(editor)
        let scroll=GameRuleCodeEditor.makeScrollView(label:"Undo test")
        let text=scroll.documentView as! NSTextView
        coordinator.attach(text)
        let theme=AppearanceThemePolicy(preferences:.factory,colorScheme:.dark,increasedContrast:false,reduceTransparency:false)
        coordinator.update(scroll,theme:theme)
        let other=GameRuleCodeEditor.makeScrollView(label:"Other buffer").documentView as! NSTextView
        XCTAssertNotNil(text.undoManager)
        XCTAssertFalse(text.undoManager === other.undoManager)
        text.insertText(" + 1",replacementRange:NSRange(location:text.string.utf16.count,length:0))
        text.breakUndoCoalescing()
        XCTAssertEqual(value,"state.remaining + 1")
        XCTAssertTrue(text.undoManager?.canUndo == true)
        let light=AppearanceThemePolicy(preferences:.factory,colorScheme:.light,increasedContrast:false,reduceTransparency:false)
        coordinator.update(scroll,theme:light)
        text.undoManager?.undo()
        XCTAssertEqual(text.string,"state.remaining")
        XCTAssertEqual(value,"state.remaining")
        text.undoManager?.redo()
        XCTAssertEqual(text.string,"state.remaining + 1")
        XCTAssertEqual(value,"state.remaining + 1")
        coordinator.detach()
    }

}
