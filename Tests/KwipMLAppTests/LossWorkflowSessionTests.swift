import XCTest
import Combine
import Darwin
import KwipMLCore
@testable import KwipMLApp

@MainActor final class LossWorkflowSessionTests:XCTestCase {
    private final class Reservation {
        var acquired=0,released=0
        var value:WorkflowExecutionReservation {.init(acquire:{[self] in acquired+=1;return UUID()},release:{[self] _ in released+=1})}
    }
    func testActualPreparedMSESupervisedSessionPublishesTypedResultsWithoutAccuracy()async throws {try await checkActualMSE(budgeted:true)}
    func testActualPreparedMSELegacyLaunchSessionPublishesTypedResultsWithoutAccuracy()async throws {try await checkActualMSE(budgeted:false)}
    func testActualWorkerLiveContradictionsRefusePublicationAndReleaseOwnership()async throws {
        for fault in ["started","epoch","completed"] {try await checkActualMSE(budgeted:true,fault:fault)}
    }
    private func checkActualMSE(budgeted:Bool,fault:String?=nil)async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Enable actual owned objective worker")}
        guard let canonical=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        defer {free(canonical)}
        let repo=URL(fileURLWithPath:development),root=URL(fileURLWithPath:String(cString:canonical)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let document=try WorkflowExamples.preparedLossObjective(task:.mse,device:"cpu")
        let library=ProjectLibrary(directory:root.appendingPathComponent("projects"));_ = try library.saveSource(.workflow(document),id:document.projectID)
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:repo,allowDevelopment:true)
        let discovery=WorkspaceStore(rootOverride:repo,storageOverride:root.appendingPathComponent("runtime-discovery"),detectCapabilities:false)
        await verifyRuntimeForIntegration(discovery)
        XCTAssertEqual(discovery.runtime,runtime)
        let composition=try XCTUnwrap(discovery.moduleCompositionDiscoveryEvidence,"Fresh objective capture requires actual selected-worker discovery")
        let registry=try JSONDecoder().decode(ComponentRegistry.self,from:Data(contentsOf:runtime.registryURL))
        let data=try DataFlowOperatorRegistry.load(from:runtime.workingDirectoryURL.appendingPathComponent("runner/dataflow-operators.json"))
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:registry,dataRegistry:data)
        let budget=try ResolvedExecutionBudget.resolve(global:.init(maximumRunSeconds:60),projectMaximumRunSeconds:nil,resources:.init(cpuThreads:2,maximumResidentBytes:1_073_741_824))
        let request=try WorkflowExecutionRequest.capture(from:owner,train:WorkflowExamples.trainAddress,runtime:runtime,executionBudget:budgeted ? budget:nil,sampleCapture:budgeted ? .factory:nil,moduleCompositionEvidence:composition)
        XCTAssertEqual(request.plan.lossRecipe?.configuration.componentID,"loss_mse");XCTAssertNil(request.plan.targetColumn)
        let source=repo.appendingPathComponent("Sources/KwipMLApp/Resources/loss-mse-observations.csv")
        let directoryID=UUID(),directory=root.appendingPathComponent("workflow-runs/\(document.projectID)/\(directoryID.uuidString)")
        let reservation=Reservation(),finished=expectation(description:"Real objective Session terminates")
        var outcome:WorkflowExecutionOutcome?
        var injected=0
        let launcher:WorkflowWorkerLauncher={executable,arguments,working,line,exit in
            try WorkflowExecutionSession.launchWorker(executable,arguments,working,{raw in
                var delivered=raw
                // These are deliberate contradictions of a real worker frame,
                // not synthetic execution evidence or a replacement worker.
                if let fault,let bytes=raw.data(using:.utf8),var frame=(try? JSONSerialization.jsonObject(with:bytes)) as? [String:Any],frame["event"] as? String == fault,injected == 0 {
                    if fault == "started",frame["trainingSamples"] != nil {frame["trainingSamples"]=999;injected+=1}
                    else if fault == "epoch",let value=frame["trainLoss"] as? Double {frame["trainLoss"]=value+10;injected+=1}
                    else if fault == "completed",let value=frame["validationLoss"] as? Double {frame["validationLoss"]=value+10;injected+=1}
                    if injected>0,let changed=try? JSONSerialization.data(withJSONObject:frame) {delivered=String(decoding:changed,as:UTF8.self)}
                }
                line(delivered)
            },exit)
        }
        let run=WorkflowExecutionSession(request:request,owner:owner,sources:[WorkflowExamples.sourceAddress:try .capture(source)],directory:directory,reservation:reservation.value,launch:launcher,onTerminal:{outcome=$0;finished.fulfill()})
        var observed:[EpochMetric]=[]
        let live=run.$liveMetrics.sink {observed.append(contentsOf:$0)}
        defer {live.cancel();if owner.isBusy {run.cancel()}}
        XCTAssertTrue(run.start())
        await fulfillment(of:[finished],timeout:65)
        if owner.isBusy {run.cancel()}
        if fault != nil {
            XCTAssertEqual(injected,1,"A real training frame must actually be contradicted")
            XCTAssertEqual(run.phase,.failed)
            XCTAssertEqual(reservation.acquired,1);XCTAssertEqual(reservation.released,1);XCTAssertFalse(owner.isBusy)
            XCTAssertNil(outcome?.recordError);XCTAssertNotNil(outcome?.error)
            XCTAssertNil(run.objectivePredictionPreview);XCTAssertNil(run.objectiveEvaluation);XCTAssertNil(run.modelCaptures)
            XCTAssertNil(outcome?.modelManifestSHA256)
            return
        }
        XCTAssertEqual(run.phase,.completed,run.statusMessage+" / "+(outcome?.error ?? ""))
        XCTAssertEqual(reservation.acquired,1);XCTAssertEqual(reservation.released,1);XCTAssertFalse(owner.isBusy)
        XCTAssertNil(outcome?.recordError)
        XCTAssertEqual(run.metrics.count,request.plan.settings.epochs)
        XCTAssertEqual(run.metrics.map(\.epoch),run.liveMetrics.map(\.epoch))
        XCTAssertTrue(run.metrics.allSatisfy {$0.accuracy == nil && $0.trainLoss.isFinite && $0.validationLoss.isFinite})
        XCTAssertTrue(observed.allSatisfy {$0.accuracy == nil});XCTAssertFalse(observed.isEmpty)
        XCTAssertEqual(run.metrics.map(\.trainLoss),run.liveMetrics.map(\.trainLoss));XCTAssertEqual(run.metrics.map(\.validationLoss),run.liveMetrics.map(\.validationLoss))
        XCTAssertNil(run.predictionPreview)
        XCTAssertNotNil(run.objectivePredictionPreview);XCTAssertEqual(run.objectiveEvaluation?.componentID,"loss_mse")
        let captures=try XCTUnwrap(run.modelCaptures)
        for origin in request.plan.model.origins.values {XCTAssertNotNil(captures.ports(instancePath:origin.instancePath,localNodeID:origin.localNodeID))}
        let result=try XCTUnwrap(outcome);XCTAssertNotNil(result.modelManifestSHA256);XCTAssertNotNil(result.preparationManifestSHA256)
        let reference=WorkflowHistoryReference(projectID:document.projectID,directoryID:directoryID)
        guard case .verified(let cold)=WorkflowHistoryReader.load(reference,root:root),let content=cold.modelContent else {return XCTFail("Completed typed objective archive must reopen independently")}
        XCTAssertEqual(cold.outcome.status,.completed);XCTAssertEqual(content.metrics.map(\.epoch),run.metrics.map(\.epoch));XCTAssertTrue(content.metrics.allSatisfy {$0.accuracy == nil})
        XCTAssertEqual(content.objectiveEvaluation?.componentID,"loss_mse");XCTAssertNotNil(content.objectivePredictionPreview);XCTAssertNotNil(content.objectiveModelCode)
        XCTAssertEqual(content.objectiveContext?.recipe.configuration,request.plan.lossRecipe?.configuration)
    }
}
