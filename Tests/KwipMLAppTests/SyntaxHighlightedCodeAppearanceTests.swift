import XCTest
import AppKit
import SwiftUI
import KwipMLCore
@testable import KwipMLApp

@MainActor final class SyntaxHighlightedCodeAppearanceTests: XCTestCase {
    private let dark=AppearanceThemePolicy(preferences:.factory,colorScheme:.dark,increasedContrast:false,reduceTransparency:false)
    private let light=AppearanceThemePolicy(preferences:.init(mode:.light,palette:.teal),colorScheme:.light,increasedContrast:false,reduceTransparency:false)
    func testThemeOnlyUpdatePreservesActualNativeSelectionScrollAndSource() throws {
        let source=(0..<250).map {"def layer_\($0)(x): return x + 1 # café"}.joined(separator:"\n")
        let scroll=SyntaxHighlightedCode.makeScrollView(theme:dark)
        scroll.frame=NSRect(x:0,y:0,width:360,height:160)
        let coordinator=SyntaxHighlightedCode.Coordinator()
        coordinator.update(scroll,source:source,theme:dark)
        let text=try XCTUnwrap(scroll.documentView as? NSTextView)
        let selected=[NSValue(range:NSRange(location:100,length:12))]
        text.selectedRanges=selected
        scroll.contentView.scroll(to:NSPoint(x:20,y:450))
        scroll.reflectScrolledClipView(scroll.contentView)
        let origin=scroll.contentView.bounds.origin
        XCTAssertGreaterThan(origin.y,0)
        coordinator.update(scroll,source:source,theme:light)
        XCTAssertEqual(text.string,source)
        XCTAssertEqual(text.selectedRanges,selected)
        XCTAssertEqual(scroll.contentView.bounds.origin,origin)
        XCTAssertEqual(text.backgroundColor,light.native.background)
        XCTAssertEqual(scroll.backgroundColor,light.native.background)
        XCTAssertEqual(text.insertionPointColor,light.native.text)
        XCTAssertEqual(text.textStorage?.attribute(.foregroundColor,at:0,effectiveRange:nil) as? NSColor,light.native.keyword)
        coordinator.update(scroll,source:source,theme:light)
        XCTAssertEqual(scroll.contentView.bounds.origin,origin)
        XCTAssertEqual(text.selectedRanges,selected)
    }
    func testNewSourceStillResetsViewportToBeginning() throws {
        let scroll=SyntaxHighlightedCode.makeScrollView(theme:dark)
        scroll.frame=NSRect(x:0,y:0,width:360,height:160)
        let coordinator=SyntaxHighlightedCode.Coordinator()
        coordinator.update(scroll,source:String(repeating:"return 1\n",count:250),theme:dark)
        scroll.contentView.scroll(to:NSPoint(x:0,y:400))
        XCTAssertGreaterThan(scroll.contentView.bounds.origin.y,0)
        let next=String(repeating:"return 2\n",count:250)
        coordinator.update(scroll,source:next,theme:light)
        XCTAssertEqual((scroll.documentView as? NSTextView)?.string,next)
        XCTAssertEqual(scroll.contentView.bounds.origin.y,0,accuracy:0.01)
    }
    func testAttributedPaletteChangesTokensWithoutChangingBytesOrRanges() throws {
        let source="def f(x): return 'café' + 2 # note"
        let tokens=PythonSyntax.tokens(in:source)
        let a=PythonSyntax.attributed(source,palette:dark.native),b=PythonSyntax.attributed(source,palette:light.native)
        XCTAssertEqual(a.string,source);XCTAssertEqual(b.string,source)
        XCTAssertEqual(PythonSyntax.attributed(source).string,source)
        XCTAssertEqual(PythonSyntax.tokens(in:source),tokens)
        for token in tokens {
            let expected:NSColor
            switch token.kind {
            case .keyword:expected=light.native.keyword
            case .string:expected=light.native.string
            case .comment:expected=light.native.comment
            case .number:expected=light.native.number
            case .name:expected=light.native.name
            }
            XCTAssertEqual(b.attribute(.foregroundColor,at:token.range.location,effectiveRange:nil) as? NSColor,expected)
        }
        XCTAssertNotEqual(a.attribute(.foregroundColor,at:0,effectiveRange:nil) as? NSColor,b.attribute(.foregroundColor,at:0,effectiveRange:nil) as? NSColor)
    }
}
