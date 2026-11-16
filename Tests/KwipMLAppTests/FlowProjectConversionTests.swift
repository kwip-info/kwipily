import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class FlowProjectConversionTests: XCTestCase {
    func testActiveCutoverKeepsIdentityBindingsAndAuthoritativeSave() throws {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: storage) }
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: storage, detectCapabilities: false)
        XCTAssertTrue(workspace.loadSeparableConv2dWorkflowExample(device: "cpu"), workspace.statusMessage)
        let previous = try XCTUnwrap(workspace.workflowSession)
        let id = previous.owner.projectID, legacy = try workspace.projectLibrary.documentURL(id: previous.owner.projectID)
        let original = try Data(contentsOf: legacy)
        let sourceBinding = previous.bindings.path(for: WorkflowExamples.sourceAddress)
        XCTAssertTrue(workspace.convertActiveWorkflowToFiles(), workspace.statusMessage)
        let current = try XCTUnwrap(workspace.workflowSession)
        XCTAssertFalse(previous === current); XCTAssertTrue(previous.isRetired)
        XCTAssertEqual(current.owner.projectID, id)
        XCTAssertEqual(current.bindings.path(for: WorkflowExamples.sourceAddress), sourceBinding)
        XCTAssertNotNil(current.owner.flowRootURL)
        let block = try XCTUnwrap(current.owner.document.nodes.first { if case .modelBlock = $0.payload { return true }; return false })
        var opened: (URL, FlowDocumentRevision)?
        workspace.workflowChildren?.presentFlow = { url, pin in opened = (url, pin); return true }
        XCTAssertTrue(workspace.workflowChildren?.open(address: block.address, lease: current.owner.lease()) == true)
        XCTAssertNotNil(opened)
        XCTAssertTrue(current.owner.document.drafts.isEmpty)
        XCTAssertNil(workspace.workflowChildren?.session)

        XCTAssertTrue(current.owner.rename("File-backed root", lease: current.owner.lease()), current.owner.error ?? "")
        XCTAssertTrue(workspace.saveManagedProject(), workspace.statusMessage)
        XCTAssertEqual(workspace.projectURL, current.owner.flowRootURL)
        XCTAssertEqual(try Data(contentsOf: legacy), original)
        XCTAssertEqual(try FlowDocumentStore.snapshot(at: XCTUnwrap(current.owner.flowRootURL)).workflow(), current.owner.document)
        XCTAssertTrue(workspace.convertActiveWorkflowToFiles(), workspace.statusMessage)
        XCTAssertEqual(workspace.activeProjectID, id)
    }
}
