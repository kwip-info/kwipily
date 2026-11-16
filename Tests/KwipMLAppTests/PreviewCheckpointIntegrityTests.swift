import XCTest
import KwipMLCore
@testable import KwipMLApp
final class PreviewCheckpointIntegrityTests:XCTestCase {
    @MainActor func testFreshPreviewRejectsChangedGraphManifestHashAndFiniteWeights() async throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer{try? FileManager.default.removeItem(at:storage)}
        let store=WorkspaceStore(rootOverride:root,storageOverride:storage,detectCapabilities:false)
        let ready=await withCheckedContinuation{c in store.refreshCapabilities{c.resume(returning:$0)}}
        XCTAssertTrue(ready);store.loadExample();store.setExecution{$0.device="cpu";$0.epochs=1};store.run()
        while store.isRunning{try await Task.sleep(nanoseconds:25_000_000)}
        let directory=try XCTUnwrap(store.outputDirectory,store.runError ?? "No run")
        let graphURL=directory.appendingPathComponent("graph.json"),manifestURL=directory.appendingPathComponent("manifest.json"),checkpoint=directory.appendingPathComponent("checkpoint.pt")
        let graphBytes=try Data(contentsOf:graphURL),manifestBytes=try Data(contentsOf:manifestURL)
        var graph=try JSONDecoder().decode(GraphDocument.self,from:graphBytes);graph.name="Altered artifact"
        try JSONEncoder().encode(graph).write(to:graphURL)
        var previewLaunches = 0
        store.previewWorkerLauncher = { executable, arguments, directory, line, exit in
            previewLaunches += 1
            return try WorkflowExecutionSession.launchWorker(executable, arguments, directory, line, exit)
        }
        store.selectNode("output");store.previewSelected()
        while store.isPreviewing{try await Task.sleep(nanoseconds:25_000_000)}
        XCTAssertEqual(previewLaunches, 0, "Changed graph must be rejected before a worker launch")
        try graphBytes.write(to:graphURL)
        var manifest=try XCTUnwrap(JSONSerialization.jsonObject(with:manifestBytes) as? [String:Any])
        var metadata=try XCTUnwrap(manifest["checkpoint"] as? [String:Any]);metadata["sha256"]=String(repeating:"0",count:64);manifest["checkpoint"]=metadata
        try JSONSerialization.data(withJSONObject:manifest).write(to:manifestURL)
        store.previewSelected()
        while store.isPreviewing{try await Task.sleep(nanoseconds:25_000_000)}
        XCTAssertEqual(previewLaunches, 0, "Manifest hash mismatch must be rejected before worker launch")
        try manifestBytes.write(to:manifestURL)
        let process=Process();process.executableURL=root.appendingPathComponent(".venv/bin/python")
        process.arguments=["-c","import sys,torch; p=sys.argv[1]; s=torch.load(p,weights_only=True); s['modelState']={k:torch.zeros_like(v) for k,v in s['modelState'].items()}; torch.save(s,p)",checkpoint.path]
        try process.run();process.waitUntilExit();XCTAssertEqual(process.terminationStatus,0)
        store.previewSelected()
        while store.isPreviewing{try await Task.sleep(nanoseconds:25_000_000)}
        XCTAssertEqual(previewLaunches, 0, "Finite valid tensors are not proof of the completed run's weights")
    }
}
