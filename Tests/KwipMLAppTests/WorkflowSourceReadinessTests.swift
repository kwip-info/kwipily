import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowSourceReadinessTests:XCTestCase {
    func testMissingSourceAppearsOnComponentAndClearsFromPrivateBindingWithoutEditingGraph()throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        let document=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        let source=try XCTUnwrap(host.owner.document.nodes.first(where:DataFlowSourceOperators.isSource))
        let display=try XCTUnwrap(host.canvas.projection.displayIDByAddress[source.address])
        let before=host.owner.document
        XCTAssertTrue(host.canvas.canvasNodeFeedback(for:display).isMisconfigured)
        XCTAssertEqual(host.canvas.currentRunConfigurationFinding?.address,source.address)
        XCTAssertNotNil(host.canvas.configurationIssue(for:source.address))
        XCTAssertTrue(host.canvas.validation.issues.contains {$0.nodeID==display && $0.message.contains("Choose")})
        XCTAssertTrue(host.bindings.setPath("/private/synthetic.csv",for:source.address,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        XCTAssertFalse(host.canvas.canvasNodeFeedback(for:display).isMisconfigured,"Binding existence is a cached readiness check; it must not stat or read the source on render.")
        XCTAssertNil(host.canvas.configurationIssue(for:source.address))
        XCTAssertEqual(host.owner.document,before)
        XCTAssertTrue(host.bindings.setPath(nil,for:source.address,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        XCTAssertTrue(host.canvas.canvasNodeFeedback(for:display).isMisconfigured)
    }
}
