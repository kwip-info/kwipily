import XCTest
import Combine
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class WorkflowFrozenModelSnapshotTests:XCTestCase {
    @MainActor func testVerifiedTemporalParentStagesWithoutLiveSourceAndSurvivesArchiveRemoval()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Requires actual training runtime")}
        let canonical=try XCTUnwrap(realpath(FileManager.default.temporaryDirectory.path,nil));defer {free(canonical)}
        let root=URL(fileURLWithPath:String(cString:canonical)).appendingPathComponent(UUID().uuidString)
        let storage=root.appendingPathComponent("workspace")
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:storage,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadTemporalSplitWorkflowExample(device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        defer {if workspace.isBusy {host.cancel()};if !workspace.isBusy {try? FileManager.default.removeItem(at:root)}}
        let source=try XCTUnwrap(host.bindings.path(for:WorkflowExamples.sourceAddress))
        let done=expectation(description:"Real temporal training")
        let observation=host.$outcomes.filter{!$0.isEmpty}.prefix(1).sink{_ in done.fulfill()}
        defer {observation.cancel()}
        guard host.run() else {return XCTFail(host.statusMessage)}
        await fulfillment(of:[done],timeout:60)
        let outcome=try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(outcome.status,.completed,outcome.error ?? "")
        let history=WorkflowHistoryReference(projectID:host.owner.projectID,directoryID:try XCTUnwrap(UUID(uuidString:outcome.directory.lastPathComponent)))
        let selected=try WorkflowFrozenModelSnapshot.select(history,root:storage)
        let originalCheckpoint=try Data(contentsOf:outcome.directory.appendingPathComponent("training/artifacts/checkpoint.pt"))
        var document=try WorkflowCodec.migrateToVersion5(host.owner.document)
        let frozen=WorkflowNodeAddress(domain:.control,localID:"saved"),evaluate=WorkflowNodeAddress(domain:.control,localID:"final")
        document.nodes += [.init(address:frozen,payload:.frozenModel(config:[:])),.init(address:evaluate,payload:.evaluateTest(config:["device":.string("cpu"),"batchSize":7]))]
        document.edges += [.init(from:.init(address:frozen,port:"model"),to:.init(address:evaluate,port:"model"),kind:.frozenModel),
                           .init(from:.init(address:.init(domain:.data,localID:"split"),port:"test"),to:.init(address:evaluate,port:"test"),kind:.table)]
        let plan=try WorkflowEvaluationCompiler.compile(document,evaluate:evaluate,modelRegistry:host.owner.modelRegistry,dataRegistry:host.owner.dataRegistry)
        try FileManager.default.removeItem(atPath:source)
        _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        let evaluationOwner=try WorkflowAuthoringStore(library:workspace.projectLibrary,projectID:document.projectID,modelRegistry:host.owner.modelRegistry,dataRegistry:host.owner.dataRegistry)
        let bindings=try WorkflowPrivateFrozenModelBindings(owner:evaluationOwner,directory:storage)
        XCTAssertTrue(bindings.setReference(selected,for:frozen,lease:evaluationOwner.lease(),expectedRevision:0))
        let policy=try ResolvedExecutionBudget.resolve(global:.init(maximumRunSeconds:60),projectMaximumRunSeconds:nil)
        let request=try await WorkflowEvaluationRequest.capture(from:evaluationOwner,bindings:bindings,evaluate:evaluate,runtime:try XCTUnwrap(workspace.runtime),root:storage,executionBudget:policy)
        XCTAssertTrue(request.isCurrent(in:evaluationOwner,bindings:bindings))
        try request.verifyCapturedInputs()
        let snapshot=request.snapshot
        defer {try? FileManager.default.removeItem(at:snapshot.directory)}
        XCTAssertEqual(snapshot.reference,selected)
        XCTAssertEqual(try Data(contentsOf:outcome.directory.appendingPathComponent("training/artifacts/checkpoint.pt")),originalCheckpoint)
        XCTAssertFalse(snapshot.files.contains{$0.path.hasPrefix("inputs/")})
        XCTAssertTrue(snapshot.files.contains{$0.path == "preparation/artifacts/dataflow.json"})
        XCTAssertTrue(snapshot.files.contains{$0.path == "training/artifacts/checkpoint.pt"})
        XCTAssertNoThrow(try snapshot.verify())
        let stagedCheckpoint=snapshot.directory.appendingPathComponent("training/artifacts/checkpoint.pt")
        try Data("tampered".utf8).write(to:stagedCheckpoint)
        XCTAssertThrowsError(try snapshot.verify())
        try originalCheckpoint.write(to:stagedCheckpoint)
        XCTAssertNoThrow(try snapshot.verify())
        try FileManager.default.removeItem(at:stagedCheckpoint)
        try FileManager.default.createSymbolicLink(at:stagedCheckpoint,withDestinationURL:outcome.directory.appendingPathComponent("training/artifacts/checkpoint.pt"))
        XCTAssertThrowsError(try snapshot.verify())
        try FileManager.default.removeItem(at:stagedCheckpoint)
        try originalCheckpoint.write(to:stagedCheckpoint)
        let receiptURL=snapshot.directory.appendingPathComponent("snapshot.json")
        let receiptBytes=try Data(contentsOf:receiptURL)
        try Data("{}".utf8).write(to:receiptURL)
        XCTAssertThrowsError(try snapshot.verify())
        try receiptBytes.write(to:receiptURL)
        // Source and split declarations cannot silently select old archived data.
        var changed=document
        let split=try XCTUnwrap(changed.nodes.firstIndex{$0.address.localID == "split"})
        if case .dataOperator(let id,var config)=changed.nodes[split].payload {
            config["gap"] = .integer(1);changed.nodes[split].payload = .dataOperator(operatorID:id,config:config)
        }
        XCTAssertThrowsError(try WorkflowFrozenModelSnapshot.stage(selected,document:changed,plan:plan,root:storage))
        let manifestURL=outcome.directory.appendingPathComponent("training/artifacts/manifest.json")
        let originalManifest=try Data(contentsOf:manifestURL)
        try Data("{}".utf8).write(to:manifestURL)
        XCTAssertThrowsError(try WorkflowFrozenModelSnapshot.stage(selected,document:document,plan:plan,root:storage))
        try originalManifest.write(to:manifestURL)
        let supervisedDone=expectation(description:"Supervised native frozen evaluation")
        var supervisedResult:WorkflowEvaluationSessionResult?,released=0
        let admittedRequest = try await WorkflowEvaluationRequest.capture(from: evaluationOwner, bindings: bindings,
            evaluate: evaluate, runtime: try XCTUnwrap(workspace.runtime), root: storage, executionBudget: policy,
            storageLimitBytes: 2_000_000_000)
        XCTAssertTrue(admittedRequest.snapshot.directory.path.contains("/evaluation-staging/"))
        let storageAdmission = try WorkflowStorageAdmission(workspace: storage)
        let workerExited = expectation(description: "Admitted evaluation waits for terminal delivery")
        var pendingEvaluationExit: (() -> Void)?
        let supervised=WorkflowEvaluationSession(request:admittedRequest,owner:evaluationOwner,bindings:bindings,workspace:storage,
            reservation:.init(acquire:{UUID()},release:{_ in released+=1}), launch: { executable, arguments, directory, line, exit in
                try WorkflowExecutionSession.launchWorker(executable, arguments, directory, line, { code, message in
                    pendingEvaluationExit = { exit(code, message) }; workerExited.fulfill()
                })
            },onTerminal:{supervisedResult=$0;supervisedDone.fulfill()})
        XCTAssertTrue(supervised.start())
        await fulfillment(of:[workerExited],timeout:45)
        XCTAssertNil(supervisedResult)
        XCTAssertGreaterThan(try storageAdmission.snapshot().reservedGrowthBytes, 0)
        XCTAssertTrue(FileManager.default.fileExists(atPath: admittedRequest.snapshot.directory.path))
        try XCTUnwrap(pendingEvaluationExit)()
        await fulfillment(of:[supervisedDone],timeout:45)
        XCTAssertEqual(try storageAdmission.snapshot().reservedGrowthBytes, 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: admittedRequest.snapshot.directory.path))
        let result=try XCTUnwrap(supervisedResult)
        guard result.outcome.status == .completed,result.recordError==nil else {return XCTFail(result.outcome.error ?? result.recordError ?? supervised.statusMessage)}
        XCTAssertNil(result.exposureError);XCTAssertEqual(released,1);XCTAssertFalse(evaluationOwner.isBusy)
        XCTAssertEqual(try WorkflowEvaluationOwnership.probe(directory:result.directory,request:try XCTUnwrap(supervised.workerInput).request),.inactive)
        XCTAssertEqual(try Data(contentsOf:outcome.directory.appendingPathComponent("training/artifacts/checkpoint.pt")),originalCheckpoint)
        let historyReference=WorkflowEvaluationHistoryReference(projectID:document.projectID,evaluationID:supervised.evaluationID)
        guard case .terminal(let reopened)=WorkflowEvaluationHistoryReader.load(historyReference,root:storage) else {return XCTFail("Completed evaluation did not reopen")}
        XCTAssertFalse(reopened.captureIncomplete)
        XCTAssertEqual(reopened.result?.manifestSHA256,result.outcome.manifestSHA256)
        try FileManager.default.removeItem(at:outcome.directory)
        guard case .terminal(let independent)=WorkflowEvaluationHistoryReader.load(historyReference,root:storage) else {return XCTFail("Evaluation history depended on deleted parent archive")}
        XCTAssertEqual(independent.result?.summary.evaluatedCount,reopened.result?.summary.evaluatedCount)
        let summaryFile=result.directory.appendingPathComponent("artifacts/summary.json")
        let savedSummary=try Data(contentsOf:summaryFile)
        try Data("{}".utf8).write(to:summaryFile)
        guard case .unavailable=WorkflowEvaluationHistoryReader.load(historyReference,root:storage) else {return XCTFail("Tampered evaluation reopened")}
        try savedSummary.write(to:summaryFile)
        let raw=try XCTUnwrap(independent.result)
        guard case .string(let sourceHash)=raw.manifest.testProvenance["sourceHash"] else {return XCTFail("Missing source identity")}
        let population=try WorkflowEvaluationExposureStore.Population(checkpointSHA256:raw.manifest.parent.checkpointSHA256,sourceSHA256:sourceHash,observationIDs:raw.predictions.map(\.observationID))
        let exposure=WorkflowEvaluationExposureStore(root:storage)
        guard case .tracked(let evaluated)=exposure.history(for:population) else {return XCTFail("Session did not durably record evaluation")}
        XCTAssertEqual(evaluated.evaluatedCount,1);XCTAssertEqual(evaluated.inspectionCount,0)
        let disclosed=try WorkflowEvaluationDisclosure.load(historyReference,root:storage)
        XCTAssertEqual(disclosed.value.manifestSHA256,raw.manifestSHA256)
        XCTAssertEqual(disclosed.priorExposure.inspectionCount,0)
        _=try WorkflowEvaluationDisclosure.load(historyReference,root:storage)
        guard case .tracked(let inspected)=exposure.history(for:population) else {return XCTFail("Disclosure was not recorded")}
        XCTAssertEqual(inspected.evaluatedCount,1);XCTAssertEqual(inspected.inspectionCount,1)
        // Exercise the native export writer with actual supervision, after both the
        // original CSV and training archive have gone away. Each export owns a new stage.
        guard case .string(let stateDigest)=raw.manifest.frozenLineage["stateDigest"] else {return XCTFail("Missing frozen state digest")}
        let exportIdentity=WorkflowEvaluationExportIdentity(evaluationID:historyReference.evaluationID.uuidString,
            evaluationManifestSHA256:raw.manifestSHA256,checkpointSHA256:raw.manifest.parent.checkpointSHA256,
            stateDigest:stateDigest,snapshotReceiptSHA256:raw.manifest.snapshotReceiptSHA256)
        let exportRuntime=try XCTUnwrap(workspace.runtime)
        let writer=WorkflowEvaluationExportWriter(storageLimitBytes: 2_000_000_000, launch:{executable,arguments,directory,line,exit in
            XCTAssertTrue(arguments.contains("--kwip-export-supervise"))
            XCTAssertTrue(arguments.contains("--kwip-launch-budget-sha256"))
            XCTAssertThrowsError(try WorkflowEvaluationOwnership.acquireInspection(directory:result.directory,
                projectID:historyReference.projectID,evaluationID:historyReference.evaluationID.uuidString))
            return try WorkflowExecutionSession.launchWorker(executable,arguments,directory,line,exit)
        })
        let exportHistory = WorkflowEvaluationHistoryStore(root: storage, projectID: historyReference.projectID, isCurrent: { true })
        exportHistory.workspace = workspace
        exportHistory.select(historyReference)
        let selectDeadline = Date().addingTimeInterval(10)
        while exportHistory.isLoading && Date() < selectDeadline { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertNotNil(exportHistory.disclosure)
        XCTAssertTrue(workspace.setExecutionPreferences(try .init(managedStorageLimitBytes: 1)))
        XCTAssertTrue(exportHistory.prepareExport())
        let refusedReview = try XCTUnwrap(exportHistory.export)
        let refusalDeadline = Date().addingTimeInterval(20)
        while refusedReview.isBusy && Date() < refusalDeadline { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertNil(refusedReview.package)
        XCTAssertTrue(refusedReview.statusMessage.lowercased().contains("storage"), refusedReview.statusMessage)
        exportHistory.closeExport()
        XCTAssertTrue(workspace.setExecutionPreferences(try .init(maximumRunSeconds: 120, managedStorageLimitBytes: 2_000_000_000)))
        XCTAssertTrue(workspace.setProjectExecutionOverrides(try .init(maximumRunSeconds: 90, cpuThreads: 2), projectID: historyReference.projectID))
        XCTAssertTrue(exportHistory.prepareExport())
        XCTAssertTrue(workspace.resetProjectExecutionOverrides(projectID: historyReference.projectID))
        let acceptedReview = try XCTUnwrap(exportHistory.export)
        let acceptedDeadline = Date().addingTimeInterval(30)
        while acceptedReview.isBusy && Date() < acceptedDeadline { try await Task.sleep(for: .milliseconds(10)) }
        let acceptedPackage = try XCTUnwrap(acceptedReview.package, acceptedReview.statusMessage)
        let capturedExport = try JSONDecoder().decode(WorkflowEvaluationExportRequest.self,
            from: Data(contentsOf: acceptedPackage.directory.deletingLastPathComponent().appendingPathComponent("request.json")))
        XCTAssertEqual(capturedExport.executionBudget?.maximumRunSeconds, 90)
        XCTAssertEqual(capturedExport.executionBudget?.resources?.cpuThreads, 2)
        exportHistory.closeExport()
        XCTAssertTrue(workspace.resetExecutionPreferences())
        let refusedWriter = WorkflowEvaluationExportWriter(storageLimitBytes: 1)
        do {
            _ = try await refusedWriter.stage(reference: historyReference, root: storage, runtime: exportRuntime,
                expected: exportIdentity, includesTestRows: false)
            XCTFail("Export quota must refuse before worker staging")
        } catch { XCTAssertTrue(error is WorkflowStorageAdmission.Failure) }
        XCTAssertEqual(try storageAdmission.snapshot().reservedGrowthBytes, 0)
        let failedWriter = WorkflowEvaluationExportWriter(storageLimitBytes: 2_000_000_000,
            launch: { _, _, _, _, _ in throw WorkflowArchiveError("Injected export launch failure") })
        do {
            _ = try await failedWriter.stage(reference: historyReference, root: storage, runtime: exportRuntime,
                expected: exportIdentity, includesTestRows: false)
            XCTFail("Injected launch failure should propagate")
        } catch { XCTAssertTrue(error.localizedDescription.contains("Injected export launch failure")) }
        XCTAssertEqual(try storageAdmission.snapshot().reservedGrowthBytes, 0)
        for includeRows in [false,true] {
            let package=try await writer.stage(reference:historyReference,root:storage,runtime:exportRuntime,
                expected:exportIdentity,includesTestRows:includeRows,executionBudget:policy)
            XCTAssertGreaterThan(try storageAdmission.snapshot().reservedGrowthBytes, 0,
                "Review must retain storage admission after worker exit")
            XCTAssertEqual(package.identity,exportIdentity)
            XCTAssertEqual(package.directory,package.cleanupDirectory.appendingPathComponent("package"))
            XCTAssertNotEqual(package.cleanupDirectory,result.directory)
            XCTAssertEqual(FileManager.default.fileExists(atPath:package.directory.appendingPathComponent("test-input.json").path),includeRows)
            let destination=root.appendingPathComponent(includeRows ? "portable-with-rows":"portable-model-only")
            try await writer.publish(package,to:destination)
            XCTAssertGreaterThan(try storageAdmission.snapshot().reservedGrowthBytes, 0)
            package.remove()
            XCTAssertEqual(try storageAdmission.snapshot().reservedGrowthBytes, 0)
            XCTAssertFalse(FileManager.default.fileExists(atPath: package.cleanupDirectory.path))
            XCTAssertTrue(FileManager.default.fileExists(atPath:result.directory.path))
            if includeRows {
                let finishedReplay=expectation(description:"Relocated portable evaluation replay")
                var replayCode:Int32?,replayLines:[String]=[]
                let worker=try WorkflowExecutionSession.launchWorker(exportRuntime.executableURL,
                    [destination.appendingPathComponent("portable_evaluation.py").path,"--replay","--device","cpu","--batch-size","7"],
                    destination,{replayLines.append($0)},{code,error in replayCode=code;if !error.isEmpty {replayLines.append(error)};finishedReplay.fulfill()})
                defer {worker.cancel()}
                await fulfillment(of:[finishedReplay],timeout:45)
                XCTAssertEqual(replayCode,0,replayLines.joined(separator:"\n"))
                let replayObject=try XCTUnwrap(replayLines.compactMap{line -> [String:Any]? in
                    guard let data=line.data(using:.utf8) else {return nil};return (try? JSONSerialization.jsonObject(with:data)) as? [String:Any]
                }.last)
                XCTAssertEqual(replayObject["evaluatedCount"] as? Int,raw.summary.evaluatedCount)
            }
        }
        // A damaged exposure ledger must not destroy the successful evaluation archive,
        // release metrics, or leak the workspace reservation.
        let exposureIndex=storage.appendingPathComponent("workflow-evaluation-exposure/index.json")
        let originalIndex=try Data(contentsOf:exposureIndex)
        try Data("{".utf8).write(to:exposureIndex)
        XCTAssertThrowsError(try WorkflowEvaluationDisclosure.load(historyReference,root:storage))
        do {
            _=try await writer.stage(reference:historyReference,root:storage,runtime:exportRuntime,expected:exportIdentity,includesTestRows:false)
            XCTFail("Damaged disclosure journal permitted export")
        } catch {XCTAssertFalse(error is CancellationError)}
        let blockedDone=expectation(description:"Evaluation with unavailable disclosure tracking")
        var blockedResult:WorkflowEvaluationSessionResult?
        let blocked=WorkflowEvaluationSession(request:request,owner:evaluationOwner,bindings:bindings,workspace:storage,
            reservation:.init(acquire:{UUID()},release:{_ in released+=1}),onTerminal:{blockedResult=$0;blockedDone.fulfill()})
        XCTAssertTrue(blocked.start());await fulfillment(of:[blockedDone],timeout:45)
        let unavailable=try XCTUnwrap(blockedResult)
        XCTAssertEqual(unavailable.outcome.status,.completed);XCTAssertNil(unavailable.recordError)
        XCTAssertNotNil(unavailable.exposureError);XCTAssertEqual(released,2);XCTAssertFalse(evaluationOwner.isBusy)
        guard case .terminal=WorkflowEvaluationHistoryReader.load(.init(projectID:document.projectID,evaluationID:blocked.evaluationID),root:storage) else {return XCTFail("Tracking failure destroyed completed archive")}
        try originalIndex.write(to:exposureIndex)
        let python=try XCTUnwrap(workspace.runtime).executableURL
        let evaluationDirectory=root.appendingPathComponent(UUID().uuidString)
        let workerInput=try WorkflowEvaluationWorkerInput.publish(request:request,evaluationID:UUID(),directory:evaluationDirectory)
        let immutableRequest=try Data(contentsOf:evaluationDirectory.appendingPathComponent("request.json"))
        XCTAssertThrowsError(try WorkflowEvaluationWorkerInput.publish(request:request,evaluationID:UUID(),directory:evaluationDirectory))
        XCTAssertEqual(try Data(contentsOf:evaluationDirectory.appendingPathComponent("request.json")),immutableRequest)
        workspace.newProject()
        let relocated=root.appendingPathComponent("relocated")
        try FileManager.default.moveItem(at:snapshot.directory,to:relocated)
        try FileManager.default.removeItem(at:storage)
        let script=root.appendingPathComponent("evaluate.py")
        try #"""
        import json,sys
        from pathlib import Path
        sys.path.insert(0,sys.argv[1])
        from frozen_checkpoint import load_frozen
        from prepared_data import load_test_partition
        from final_evaluation import evaluate_classifier
        root=Path(sys.argv[2]);receipt=json.loads((root/'snapshot.json').read_text());ref=receipt['reference']
        files={f['path']:f for f in receipt['files']}
        frozen=load_frozen(root/'training/artifacts/checkpoint.pt',expected_manifest_sha256=ref['modelManifestSHA256'],expected_checkpoint_sha256=ref['checkpointSHA256'],registry_path=root/'model-registry.json',expected_registry_sha256=files['model-registry.json']['sha256'])
        (x,y),provenance,_=load_test_partition(root/'preparation/artifacts/manifest.json','split','test',frozen.training_provenance,expected_manifest_sha256=ref['preparationManifestSHA256'])
        result=evaluate_classifier(frozen.model,x,y,provenance['testRowIDs'],batch_size=7,device='cpu')
        assert result['evaluatedCount']==len(y)>0
        print('VERIFIED_FROZEN_TEST',len(y))
        """#.write(to:script,atomically:true,encoding:.utf8)
        let replay=expectation(description:"Frozen inference from private relocated snapshot")
        var code:Int32?,lines:[String]=[]
        let process=try WorkflowExecutionSession.launchWorker(python,[script.path,URL(fileURLWithPath:development).appendingPathComponent("runner").path,relocated.path],root,{lines.append($0)},{value,error in code=value;if !error.isEmpty {lines.append(error)};replay.fulfill()})
        defer {process.cancel()}
        await fulfillment(of:[replay],timeout:45)
        XCTAssertEqual(code,0,lines.joined(separator:"\n"))
        XCTAssertTrue(lines.contains{$0.contains("VERIFIED_FROZEN_TEST")},lines.joined(separator:"\n"))
        let movedRun=root.appendingPathComponent("relocated-evaluation")
        try FileManager.default.moveItem(at:evaluationDirectory,to:movedRun)
        let cliDone=expectation(description:"Actual evaluation CLI")
        var cliCode:Int32?,events:[String]=[]
        let cli=try WorkflowExecutionSession.launchWorker(python,[URL(fileURLWithPath:development).appendingPathComponent("runner/main.py").path,"--evaluate-frozen",movedRun.appendingPathComponent("request.json").path,"--evaluation-request-sha256",try workerInput.request.requestSHA256()],root,{events.append($0)},{value,error in cliCode=value;if !error.isEmpty {events.append(error)};cliDone.fulfill()})
        defer {cli.cancel()}
        await fulfillment(of:[cliDone],timeout:45)
        guard cliCode == 0 else {return XCTFail(events.joined(separator:"\n"))}
        XCTAssertTrue(events.contains{$0.contains("evaluationCompleted")},events.joined(separator:"\n"))
        let inputSnapshot=WorkflowFrozenModelSnapshot(reference:snapshot.reference,directory:movedRun.appendingPathComponent("input"),files:snapshot.files,receiptSHA256:snapshot.receiptSHA256)
        let verified=try WorkflowEvaluationArtifacts.verify(request:workerInput.request,snapshot:inputSnapshot,root:movedRun)
        XCTAssertGreaterThan(verified.summary.evaluatedCount,0)
        XCTAssertEqual(verified.predictions.count,verified.summary.evaluatedCount)
    }
}
