import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowPrivateCopyRouteTests: XCTestCase {
    func testPrivateActionAndOrdinaryResumeReachDurableEditorAndRestoreOnlyExistingDraft() throws {
        _ = NSApplication.shared
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let library = ProjectLibrary(directory: directory.appendingPathComponent("projects"))
        var source = try WorkflowExamples.preparedSeparableConv2dClassifier()
        var secondBlock = try XCTUnwrap(source.nodes.first { $0.address.localID == "convolution" })
        secondBlock.address = .init(domain: .model, localID: "second-block")
        source.nodes.append(secondBlock)
        _ = try library.saveSource(.workflow(source)); _ = try library.activateFileBackedWorkflow(id: source.projectID)
        let record = try XCTUnwrap(try library.listSources().first)
        let registry = ProjectWindowRegistry(presentsWindows: false, makeWorkspace: {
            WorkspaceStore(rootOverride: repo, storageOverride: directory, detectCapabilities: false)
        })
        let controller = try registry.open(record, present: false)
        let workspace = controller.session.workspace
        let host = try XCTUnwrap(workspace.workflowSession), children = try XCTUnwrap(workspace.workflowChildren)
        let address = WorkflowNodeAddress(domain: .model, localID: "convolution")
        var openedFiles = 0
        children.presentFlow = { _, _ in openedFiles += 1; return true }
        children.present = { session, path in session.open(path: path) }
        XCTAssertTrue(children.open(address: address, lease: host.owner.lease(), presentWindow: false))
        XCTAssertEqual(openedFiles, 1); XCTAssertTrue(host.owner.document.drafts.isEmpty)
        XCTAssertTrue(host.owner.select(.init(nodeAddresses: [address]), lease: host.owner.lease()))
        let context = try WorkflowModelNodeContext(owner: host.owner, address: address,
            onOpenBlock: host.canvas.onOpenBlock, onEditPrivateBlock: host.canvas.onEditPrivateBlock)
        XCTAssertTrue(context.editPrivateCopy())
        XCTAssertEqual(openedFiles, 1)
        let editor = try XCTUnwrap(children.session?.editor)
        let node = try XCTUnwrap(editor.scope?.nodes.first)
        editor.moveNodes(ids: [node.id], dx: 17, dy: 0)
        let edited = host.owner.document
        XCTAssertEqual(edited.drafts.first?.proposed.body.nodes.first(where: { $0.id == node.id })?.x, node.x + 17)
        XCTAssertTrue(children.open(address: address, lease: host.owner.lease(), presentWindow: false))
        XCTAssertEqual(openedFiles, 1)
        let secondAddress = secondBlock.address
        XCTAssertTrue(children.open(address: secondAddress, lease: host.owner.lease(), presentWindow: false))
        XCTAssertEqual(openedFiles, 2)
        XCTAssertTrue(host.canvas.canOpenBlock(try XCTUnwrap(host.canvas.projection.displayIDByAddress[secondAddress])))
        XCTAssertFalse(children.openPrivateCopy(address: secondAddress, lease: host.owner.lease(), presentWindow: false))
        XCTAssertEqual(host.owner.document, edited)
        XCTAssertTrue(registry.close(controller))
        let target = CanvasTabSession.Target.inlineScope(root: .managed(UUID(uuidString: source.projectID)!), kind: .workflow, path: [address.localID])
        _ = try registry.restoreCanvas(target)
        let restored = try XCTUnwrap(registry.openControllers.first)
        let restoredHost = try XCTUnwrap(restored.session.workspace.workflowSession)
        let restoredChildren = try XCTUnwrap(restored.session.workspace.workflowChildren)
        XCTAssertEqual(restoredHost.owner.document, edited)
        XCTAssertEqual(restoredChildren.session?.editor?.path, [address.localID])
        XCTAssertTrue(try XCTUnwrap(restoredChildren.session).apply())
        XCTAssertTrue(restoredHost.owner.document.drafts.isEmpty)
        XCTAssertTrue(registry.close(restored))
        XCTAssertThrowsError(try registry.restoreCanvas(target))
        let final = try XCTUnwrap(registry.openControllers.first)
        XCTAssertTrue(try XCTUnwrap(final.session.workspace.workflowSession).owner.document.drafts.isEmpty)
        XCTAssertTrue(registry.close(final))
    }
}
