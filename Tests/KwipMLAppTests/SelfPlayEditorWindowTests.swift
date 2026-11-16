import XCTest
import AppKit
import SwiftUI
@testable import KwipMLApp

@MainActor final class SelfPlayEditorWindowTests: XCTestCase {
    func testConfigurationDoesNotBeginModalSessionAndClosesOnce() throws {
        _ = NSApplication.shared
        let controller=SelfPlayEditorWindowController {_ in AnyView(Text("Editor fixture"))}
        let window=try XCTUnwrap(controller.window)
        var discarded=0
        controller.discardDraft={discarded += 1}
        XCTAssertNil(window.sheetParent)
        XCTAssertFalse(window.styleMask.contains(.docModalWindow))
        XCTAssertNil(NSApp.modalWindow,"Configuration must not take over the application event loop")
        XCTAssertTrue(window.styleMask.contains(.closable))
        XCTAssertEqual(window.tabbingMode,.disallowed)
        window.performClose(nil) // native titlebar / Cmd-W path
        XCTAssertTrue(controller.didClose)
        XCTAssertEqual(discarded,1)
        XCTAssertNil(window.contentView,"Close must release the hosted editor, not just hide it")
        controller.close()
        XCTAssertEqual(discarded,1)
    }

    func testCloseButtonAndOwnerDepartureUseSameCleanupEvenWithFocusedField() throws {
        _ = NSApplication.shared
        for nativeClose in [false,true] {
            let field=NSTextField(string:"unfinished draft")
            let controller=SelfPlayEditorWindowController {_ in AnyView(Text("Editor fixture"))}
            let window=try XCTUnwrap(controller.window)
            window.contentView=field
            XCTAssertTrue(window.makeFirstResponder(field))
            var cancelled=false
            controller.discardDraft={cancelled=true}
            if nativeClose {window.performClose(nil)} else {controller.close()}
            XCTAssertTrue(cancelled)
            XCTAssertTrue(controller.didClose)
            XCTAssertNil(window.contentView)
        }
    }
}

@MainActor final class ConfigurationDraftLifetimeTests:XCTestCase {
    func testDirtyClosePreservesDraftOnCancelAndDiscardsOnlyOnceAfterApproval()throws {
        _=NSApplication.shared
        let controller=ExpandedConfigurationWindowController{_ in AnyView(Text("Draft fixture"))}
        let window=try XCTUnwrap(controller.window)
        var draft="edited value",discarded=0,requests=0
        var answer:((Bool)->Void)?
        controller.hasUnappliedDraft={!draft.isEmpty}
        controller.discardDraft={draft="";discarded += 1}
        controller.confirmDiscard={_,completion in requests += 1;answer=completion}
        controller.close()
        controller.close() // repeated click while decision is pending must not duplicate it
        XCTAssertEqual(requests,1);XCTAssertFalse(controller.didClose)
        XCTAssertEqual(draft,"edited value");XCTAssertNotNil(window.contentView)
        try XCTUnwrap(answer)(false)
        XCTAssertFalse(controller.didClose);XCTAssertEqual(discarded,0)
        window.performClose(nil) // same policy for titlebar/Cmd-W
        XCTAssertEqual(requests,2)
        try XCTUnwrap(answer)(true)
        XCTAssertTrue(controller.didClose);XCTAssertEqual(discarded,1)
        controller.close();XCTAssertEqual(discarded,1)
    }

    func testInspectorDepartureRetainsDirtyEditorUntilDecisionAndQuitCanCancel()throws {
        _=NSApplication.shared
        weak var retained:ExpandedConfigurationWindowController?
        var discarded=false
        autoreleasepool {
            var controller:ExpandedConfigurationWindowController?=ExpandedConfigurationWindowController{_ in AnyView(Text("Owned draft"))}
            retained=controller
            controller?.hasUnappliedDraft={true}
            controller?.discardDraft={discarded=true}
            controller?.confirmDiscard={_,done in done(false)}
            controller?.close();controller=nil // inspector loses its State reference
            XCTAssertNotNil(retained);XCTAssertFalse(discarded)
            XCTAssertFalse(ExpandedConfigurationWindowController.prepareToTerminate(confirmDiscard:{false}))
            XCTAssertNotNil(retained);XCTAssertFalse(discarded)
            XCTAssertTrue(ExpandedConfigurationWindowController.prepareToTerminate(confirmDiscard:{true}))
            XCTAssertTrue(discarded)
            if let closing=retained {
                XCTAssertTrue(closing.didClose)
                XCTAssertNil(closing.window?.contentView)
                XCTAssertNil(closing.window?.windowController)
            }
        }
        // AppKit may autorelease close-notification/window-controller temporaries.
        // Still require actual controller destruction after that bounded pool drains.
        XCTAssertNil(retained)
    }

    func testAppliedDraftClosesWithoutDiscardConfirmation() {
        _=NSApplication.shared
        let controller=ExpandedConfigurationWindowController{_ in AnyView(Text("Applied fixture"))}
        controller.hasUnappliedDraft={false}
        controller.confirmDiscard={_,_ in XCTFail("An applied draft does not need another approval")}
        controller.close();XCTAssertTrue(controller.didClose)
    }
}
