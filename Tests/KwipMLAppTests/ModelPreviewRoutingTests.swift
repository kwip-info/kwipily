import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModelPreviewRoutingTests:XCTestCase {
    @MainActor private func setup()throws->(URL,WorkspaceStore,ModelAuthoringStore) {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let temp=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:temp,detectCapabilities:false)
        let id=UUID().uuidString
        var source=CompositeExamples.parallelClassifier;source.execution=ExecutionSettings(device:"cpu")
        _ = try workspace.projectLibrary.saveDocument(.blockModel(ModelAuthoringDocument(projectID:id,source:source)),id:id)
        let owner=try ModelAuthoringStore(library:workspace.projectLibrary,projectID:id)
        workspace.activeProjectID=id;workspace.modelAuthoring=owner
        return(temp,workspace,owner)
    }
    @MainActor func testPrimitiveResolutionRejectsSyntheticBlockAndPendingDraft()throws {
        let (temp,workspace,owner)=try setup();defer{try? FileManager.default.removeItem(at:temp)}
        let id=try workspace.expandedPreviewNodeID(instancePath:["features"],localNodeID:"merge")
        XCTAssertNotEqual(id,"merge")
        XCTAssertEqual(try workspace.expandedPreviewNodeID(instancePath:[],localNodeID:"input"),"input")
        XCTAssertEqual(try owner.executionSnapshot().origins[id]?.localNodeID,"merge")
        XCTAssertThrowsError(try workspace.expandedPreviewNodeID(instancePath:[],localNodeID:"features"))
        XCTAssertThrowsError(try workspace.expandedPreviewNodeID(instancePath:["other"],localNodeID:"merge"))
        XCTAssertTrue(owner.beginDraft(path:["features"],lease:owner.lease()))
        XCTAssertThrowsError(try workspace.expandedPreviewNodeID(instancePath:["features"],localNodeID:"merge"))
    }
    @MainActor func testActualNestedPreviewResumeAndSessionIsolation()async throws {
        let (temp,workspace,owner)=try setup();defer{try? FileManager.default.removeItem(at:temp)}
        let ready=await withCheckedContinuation{continuation in workspace.refreshCapabilities{continuation.resume(returning:$0)}}
        XCTAssertTrue(ready,workspace.capabilityStatus)
        XCTAssertTrue(workspace.canPreviewModelNode(instancePath:["features"],localNodeID:"merge"))
        XCTAssertFalse(workspace.canPreviewModelNode(instancePath:[],localNodeID:"features"))
        let expanded=try workspace.expandedPreviewNodeID(instancePath:["features"],localNodeID:"merge")
        workspace.previewModelNode(instancePath:["features"],localNodeID:"merge")
        try await finish(workspace)
        XCTAssertNotNil(workspace.previews[expanded],workspace.statusMessage)
        XCTAssertNil(workspace.previews["merge"])
        XCTAssertNotNil(workspace.modelPreview(instancePath:["features"],localNodeID:"merge"))
        XCTAssertTrue(workspace.canResumeModelNode(instancePath:["features"],localNodeID:"merge"))
        workspace.previewModelNode(instancePath:["features"],localNodeID:"merge",resume:true)
        try await finish(workspace)
        XCTAssertFalse(workspace.previewReused.isEmpty,workspace.statusMessage)
        XCTAssertNotNil(workspace.previews["output"])
        // Identical reopened source must not inherit the prior window/session's mapped result.
        workspace.modelAuthoring=try ModelAuthoringStore(library:workspace.projectLibrary,projectID:owner.projectID)
        XCTAssertFalse(workspace.cacheValid)
        XCTAssertNil(workspace.modelPreview(instancePath:["features"],localNodeID:"merge"))
        XCTAssertFalse(workspace.canResumeModelNode(instancePath:["features"],localNodeID:"merge"))
    }
    @MainActor func testLateCompletionDoesNotPublishIntoReplacementOwner()async throws {
        let(temp,workspace,owner)=try setup();defer{try? FileManager.default.removeItem(at:temp)}
        _ = await withCheckedContinuation{continuation in workspace.refreshCapabilities{continuation.resume(returning:$0)}}
        workspace.previewModelNode(instancePath:["features"],localNodeID:"merge")
        XCTAssertTrue(workspace.isPreviewing,workspace.statusMessage)
        workspace.modelAuthoring=try ModelAuthoringStore(library:workspace.projectLibrary,projectID:owner.projectID)
        workspace.statusMessage="New session"
        try await finish(workspace)
        XCTAssertNil(workspace.previewCachePath)
        XCTAssertEqual(workspace.statusMessage,"New session")
    }
    @MainActor private func finish(_ workspace:WorkspaceStore)async throws {
        let deadline=Date().addingTimeInterval(30)
        while workspace.isPreviewing && Date()<deadline{try await Task.sleep(nanoseconds:25_000_000)}
        XCTAssertFalse(workspace.isPreviewing)
    }
}
