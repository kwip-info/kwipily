import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class PublishedBagWorkflowTests:XCTestCase {
    func testActualPublishedExampleDiscoversRunsAndReopensWholeObservations()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Enable the actual development runtime")}
        let repository=URL(fileURLWithPath:development)
        guard let physical=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        defer {free(physical)}
        let root=URL(fileURLWithPath:String(cString:physical)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let runtime=try XCTUnwrap(workspace.runtime)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.embeddingBagRuntimeCapabilities.isComplete)
        let proof=try XCTUnwrap(workspace.moduleCompositionDiscoveryEvidence)
        let registry=try ComponentRegistry.builtIn()
        XCTAssertEqual(try JSONDecoder().decode(ComponentRegistry.self,from:Data(contentsOf:runtime.registryURL)),registry)
        let document=try PreparedWorkflowExample.component("embedding_bag").document(device:"cpu")
        XCTAssertEqual(PreparedWorkflowExample.component("embedding_bag").observationsResource,"bag-observations")
        let library=workspace.projectLibrary
        _ = try library.saveSource(.workflow(document),id:document.projectID)
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:registry,dataRegistry:.load(from:repository.appendingPathComponent("runner/dataflow-operators.json")))
        let budget=try ResolvedExecutionBudget.resolve(global:.init(maximumRunSeconds:60),projectMaximumRunSeconds:nil,resources:.init(cpuThreads:2,maximumResidentBytes:1_073_741_824))
        let request=try WorkflowExecutionRequest.capture(from:owner,train:WorkflowExamples.trainAddress,runtime:runtime,executionBudget:budget,sampleCapture:.factory,moduleCompositionEvidence:proof)
        let id=UUID(),directory=root.appendingPathComponent("workflow-runs/\(document.projectID)/\(id.uuidString)")
        let ended=expectation(description:"Published example completes")
        var outcome:WorkflowExecutionOutcome?,acquired=0,released=0
        let reservation=WorkflowExecutionReservation(acquire:{acquired+=1;return UUID()},release:{_ in released+=1})
        let run=WorkflowExecutionSession(request:request,owner:owner,sources:[WorkflowExamples.sourceAddress:try .capture(repository.appendingPathComponent("Sources/KwipMLApp/Resources/bag-observations.csv"))],directory:directory,reservation:reservation,onTerminal:{outcome=$0;ended.fulfill()})
        XCTAssertTrue(run.start())
        await fulfillment(of:[ended],timeout:65)
        defer {if owner.isBusy {run.cancel()}}
        XCTAssertEqual(run.phase,.completed,run.statusMessage+" / "+(outcome?.error ?? ""))
        XCTAssertEqual(acquired,1);XCTAssertEqual(released,1);XCTAssertFalse(owner.isBusy)
        XCTAssertNil(outcome?.error);XCTAssertNil(outcome?.recordError)
        XCTAssertNotNil(outcome?.modelManifestSHA256)
        XCTAssertEqual(run.metrics.map(\.epoch),[1,2,3])
        let preview=try XCTUnwrap(run.bagPredictionPreview)
        XCTAssertEqual(preview.evaluationCount,16);XCTAssertEqual(preview.rows.count,16)
        XCTAssertTrue(preview.rows.allSatisfy {$0.input.indices.count == 3 && $0.input.perSampleWeights == nil && $0.label != nil && $0.predicted != nil})
        XCTAssertNil(run.predictionPreview);XCTAssertNil(run.objectivePredictionPreview)
        let captures=try XCTUnwrap(run.modelCaptures)
        for origin in request.plan.model.origins.values {XCTAssertNotNil(captures.ports(instancePath:origin.instancePath,localNodeID:origin.localNodeID))}
        let cold=try XCTUnwrap(WorkflowHistoryReader.loadModelContent(.init(projectID:document.projectID,directoryID:id),root:root))
        XCTAssertEqual(cold.bagPredictionPreview,preview)
        XCTAssertEqual(cold.metrics.map(\.trainLoss),run.metrics.map(\.trainLoss))
    }
}
