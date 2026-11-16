import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowGuidedCourseLaunchTests:XCTestCase {
    @MainActor func testCourseStartsWithBoundSourceResumesProgressAndStaysOutOfOrdinaryRecents()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        workspace.startCourse()
        let host=try XCTUnwrap(workspace.workflowSession)
        XCTAssertEqual(host.owner.document.guidedBuild?.completedSteps,1)
        XCTAssertEqual(host.owner.document.nodes.count,1)
        XCTAssertNotNil(host.bindings.path(for:WorkflowExamples.sourceAddress))
        workspace.refreshProjects()
        let record=try XCTUnwrap(workspace.lessonProjects.first(where:{$0.id == host.owner.projectID}))
        XCTAssertFalse(workspace.modelProjects.contains(where:{$0.id == record.id}))
        XCTAssertNil(host.canvas.canvasRevealRequest)
        let initial=try XCTUnwrap(host.owner.document.guidedBuild)
        let contextID=UUID(),address=WorkflowExamples.sourceAddress
        XCTAssertTrue(host.owner.select(.init(nodeAddresses:[address]),lease:host.owner.lease()))
        XCTAssertTrue(host.owner.setInspectorEditPending(contextID:contextID,address:address,key:"separator",pending:true,lease:host.owner.lease()))
        XCTAssertFalse(host.advanceGuidedCourse(expected:initial))
        XCTAssertEqual(host.owner.document.guidedBuild,initial)
        XCTAssertNil(host.canvas.canvasRevealRequest)
        host.owner.clearDataEdits(contextID:contextID)
        XCTAssertTrue(host.advanceGuidedCourse(expected:initial))
        let reveal=try XCTUnwrap(host.canvas.canvasRevealRequest)
        XCTAssertEqual(reveal.nodeIDs,host.canvas.selectedNodeIDs)
        XCTAssertFalse(host.advanceGuidedCourse(expected:initial))
        XCTAssertEqual(host.canvas.canvasRevealRequest,reveal)
        XCTAssertEqual(host.owner.document.guidedBuild?.completedSteps,2)
        workspace.newProject()
        workspace.openManagedProject(record)
        let reopened=try XCTUnwrap(workspace.workflowSession)
        XCTAssertEqual(reopened.owner.document.guidedBuild?.completedSteps,2)
        XCTAssertEqual(workspace.activeSection,"Build")
        XCTAssertFalse(host.advanceGuidedCourse(expected:initial))
    }
}
