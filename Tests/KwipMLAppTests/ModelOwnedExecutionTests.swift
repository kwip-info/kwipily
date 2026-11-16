import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModelOwnedExecutionTests:XCTestCase {
    @MainActor func testOwnedModelGeneratesCodeTrainsAndKeepsCanonicalSource() async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.isExecutableFile(atPath:repository.appendingPathComponent(".venv/bin/python").path) else {throw XCTSkip("Python runtime absent")}
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
        let id=UUID().uuidString
        _=try workspace.projectLibrary.saveDocument(.blockModel(.init(projectID:id,source:CompositeExamples.parallelClassifier)))
        XCTAssertTrue(workspace.beginOwnedProject(id:id))
        await verifyRuntimeForIntegration(workspace)
        workspace.setExecution {$0.epochs=2;$0.device="cpu"}
        let owner=try XCTUnwrap(workspace.modelAuthoring),source=owner.document
        let request=try ModelExecutionRequest.capture(from:owner,registry:workspace.registry)
        workspace.refreshDraftCode()
        let codeDeadline=Date().addingTimeInterval(20)
        while workspace.isGeneratingCode && Date()<codeDeadline {try await Task.sleep(nanoseconds:20_000_000)}
        XCTAssertNil(workspace.draftCodeError)
        XCTAssertTrue(workspace.draftCode.contains("def forward"))
        workspace.run()
        XCTAssertTrue(workspace.isRunning,workspace.runError ?? workspace.statusMessage)
        XCTAssertTrue(owner.isBusy)
        XCTAssertFalse(owner.editMetadata({name,_ in name="Cannot change during run"},lease:owner.lease()))
        let deadline=Date().addingTimeInterval(30)
        while workspace.isRunning && Date()<deadline {try await Task.sleep(nanoseconds:20_000_000)}
        XCTAssertFalse(workspace.isRunning)
        XCTAssertTrue(workspace.lastCompleted,workspace.runError ?? workspace.statusMessage)
        XCTAssertFalse(owner.isBusy)
        XCTAssertEqual(owner.document,source)
        XCTAssertEqual(try workspace.projectLibrary.loadDocument(id:id),source)
        let directory=try XCTUnwrap(workspace.outputDirectory)
        let manifest=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:directory.appendingPathComponent("manifest.json"))) as? [String:Any])
        let graphData=try JSONSerialization.data(withJSONObject:try XCTUnwrap(manifest["graph"]))
        XCTAssertEqual(try JSONDecoder().decode(GraphDocument.self,from:graphData),request.graph)
        XCTAssertTrue(FileManager.default.fileExists(atPath:directory.appendingPathComponent("composite/source.json").path))
        XCTAssertFalse(workspace.experiments.isEmpty)
        XCTAssertFalse(workspace.resultsStale)
    }
}
