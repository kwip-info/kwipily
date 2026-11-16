import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class FlowCanvasSavePresentationTests:XCTestCase {
    func testRefusedCloseSaveThenSuccessfulToolbarRetryClearsOnlySaveFailure()throws {
        try exerciseRetry(initialStatus:"")
    }
    func testSaveRetryPreservesUnrelatedCanvasStatusAndCurrentFailureAuthority()throws {
        try exerciseRetry(initialStatus:"Connection inspection note.")
    }
    private func exerciseRetry(initialStatus:String)throws {
        _ = NSApplication.shared
        let directory=try canonicalBackendTemporaryDirectory()
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:directory)}
        let plan=try FlowDocumentMigration.plan(WorkflowExamples.preparedSeparableConv2dClassifier())
        let root=try FlowDocumentStore.commitMigration(plan,to:directory.appendingPathComponent("Project"))
        let child=root.deletingLastPathComponent().appendingPathComponent(try XCTUnwrap(plan.subflows.keys.first))
        let original=try Data(contentsOf:child)
        let registry=FlowCanvasWindowRegistry(),controller=try registry.open(child,present:false)
        defer {
            try? original.write(to:child,options:.atomic)
            _ = controller.documentOwner.revert()
            controller.window?.close()
        }
        controller.canvas.updateBoundary(context:controller.canvas.boundaryRevision){$0.title="Pending local title"}
        XCTAssertTrue(controller.documentOwner.isDirty)
        controller.canvas.statusMessage=initialStatus
        var foreign=try FlowDocumentCodec.decode(original)
        guard case .modelBlock(var block)=foreign.payload else {return XCTFail("Expected child block")}
        block.title="External changed head";foreign.payload = .modelBlock(block)
        try FlowDocumentCodec.encode(foreign).write(to:child,options:.atomic)
        XCTAssertFalse(controller.documentOwner.save(),"Initial toolbar save must refuse a changed head")
        let failure=try XCTUnwrap(controller.documentOwner.error)
        XCTAssertFalse(controller.handleCloseRequest{.save},"The actual close-confirmation handler must keep this editor open")
        XCTAssertTrue(controller.documentOwner.isDirty)
        XCTAssertEqual(controller.authority.error,failure,"Current failure remains visible through the observed owner")
        XCTAssertEqual(controller.canvas.statusMessage,initialStatus,"Save failure must not overwrite an independent canvas status")
        try original.write(to:child,options:.atomic)
        XCTAssertTrue(controller.documentOwner.save(),"Toolbar retry uses the same direct owner save as the native callback")
        XCTAssertFalse(controller.documentOwner.isDirty)
        XCTAssertNil(controller.documentOwner.error);XCTAssertNil(controller.authority.error)
        XCTAssertEqual(controller.canvas.statusMessage,initialStatus,"Resolved save errors must not survive in cached footer presentation")
        XCTAssertEqual(controller.authority.editStatus,"Subflow document saved. Callers retain their pinned revision.")
        XCTAssertTrue(controller.handleCloseRequest{XCTFail("Clean editor should not ask to save again");return .cancel})
    }
}
