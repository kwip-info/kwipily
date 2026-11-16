import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowCheckpointSnapshotTests: XCTestCase {
    func testCancelledStagingStopsBeforeOpeningArchive() async throws {
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        let reference=WorkflowCheckpointReference(projectID:UUID().uuidString,directoryID:UUID().uuidString,
            runID:UUID().uuidString,requestIdentity:String(repeating:"a",count:64),modelManifestSHA256:String(repeating:"b",count:64))
        let cancelled=await Task { () -> Bool in
            withUnsafeCurrentTask {$0?.cancel()}
            do {_ = try WorkflowHistoryReader.stageCheckpoint(reference,root:root);return false}
            catch is CancellationError {return true}
            catch {return false}
        }.value
        XCTAssertTrue(cancelled)
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.path))
    }
    func testVerifiedHistoryStagesIndependentCheckpointAndWorkerFineTunesIt() async throws {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else { throw XCTSkip("Enable actual runtime") }
        // Indexed View storage refuses symlinks in every path component. Foundation
        // canonicalizes Darwin temporary URLs back through /var; keep this real-worker
        // fixture in the checkout's ignored local directory instead.
        let root = URL(fileURLWithPath: development).appendingPathComponent(".local/checkpoint-fixtures/" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: URL(fileURLWithPath: development), storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadPreparedWorkflowExample(device: "cpu"))
        let host = try XCTUnwrap(workspace.workflowSession)
        let done = expectation(description: "Parent training completes")
        let observation = host.$outcomes.dropFirst().filter { !$0.isEmpty }.prefix(1).sink { _ in done.fulfill() }
        XCTAssertTrue(host.run(train: WorkflowExamples.trainAddress))
        await fulfillment(of: [done], timeout: 60); observation.cancel()
        let outcome = try XCTUnwrap(host.outcomes.first)
        let ref = WorkflowHistoryReference(projectID: host.owner.projectID, directoryID: try XCTUnwrap(UUID(uuidString: outcome.directory.lastPathComponent)))
        let selected=WorkflowCheckpointReference(projectID:ref.projectID,directoryID:ref.directoryID.uuidString,
            runID:outcome.runID.uuidString,requestIdentity:outcome.request.requestIdentity,
            modelManifestSHA256:try XCTUnwrap(outcome.modelManifestSHA256))
        let selectedParent=await host.selectCheckpointParent(reference:ref,for:WorkflowExamples.trainAddress,lease:host.owner.lease())
        XCTAssertTrue(selectedParent)
        let parentDetails=try await host.checkpointDetails(selected)
        XCTAssertEqual(parentDetails.device,"cpu")
        XCTAssertEqual(parentDetails.completedEpoch,3)
        XCTAssertEqual(host.checkpointBindings.reference(for:WorkflowExamples.trainAddress),selected)
        XCTAssertTrue(host.clearCheckpointParent(for:WorkflowExamples.trainAddress,lease:host.owner.lease()))
        XCTAssertNil(host.checkpointBindings.reference(for:WorkflowExamples.trainAddress))
        for changed in [
            WorkflowCheckpointReference(projectID:selected.projectID,directoryID:selected.directoryID,runID:UUID().uuidString,requestIdentity:selected.requestIdentity,modelManifestSHA256:selected.modelManifestSHA256),
            WorkflowCheckpointReference(projectID:selected.projectID,directoryID:selected.directoryID,runID:selected.runID,requestIdentity:String(repeating:"0",count:64),modelManifestSHA256:selected.modelManifestSHA256),
            WorkflowCheckpointReference(projectID:selected.projectID,directoryID:selected.directoryID,runID:selected.runID,requestIdentity:selected.requestIdentity,modelManifestSHA256:String(repeating:"0",count:64)),
            WorkflowCheckpointReference(projectID:"../escape",directoryID:selected.directoryID,runID:selected.runID,requestIdentity:selected.requestIdentity,modelManifestSHA256:selected.modelManifestSHA256)
        ] {XCTAssertThrowsError(try WorkflowHistoryReader.stageCheckpoint(changed,root:root))}
        let snapshot = try WorkflowHistoryReader.stageCheckpoint(selected, root: root)
        defer { try? FileManager.default.removeItem(at: snapshot.directory) }
        XCTAssertEqual(snapshot.reference, ref)
        XCTAssertEqual(snapshot.runID, outcome.runID.uuidString)
        XCTAssertEqual(Set(try FileManager.default.contentsOfDirectory(atPath: snapshot.directory.path)), ["manifest.json", "checkpoint.pt", "metrics.json"])
        let archived=root.appendingPathComponent("child-archive")
        try FileManager.default.createDirectory(at:archived,withIntermediateDirectories:false)
        let inventory=try snapshot.archive(to:archived)
        XCTAssertEqual(Set(inventory.map(\.path)),["continuation-parent/manifest.json","continuation-parent/checkpoint.pt","continuation-parent/metrics.json"])
        for file in inventory {
            let bytes=try Data(contentsOf:archived.appendingPathComponent(file.path))
            XCTAssertEqual(bytes.count,file.bytes);XCTAssertEqual(WorkflowArchiveCodec.hash(bytes),file.sha256)
        }
        XCTAssertThrowsError(try snapshot.archive(to:archived))
        let altered=root.appendingPathComponent("altered-child")
        try FileManager.default.createDirectory(at:altered,withIntermediateDirectories:false)
        let originalMetrics=try Data(contentsOf:snapshot.directory.appendingPathComponent("metrics.json"))
        try Data("changed".utf8).write(to:snapshot.directory.appendingPathComponent("metrics.json"))
        XCTAssertThrowsError(try snapshot.archive(to:altered))
        XCTAssertFalse(FileManager.default.fileExists(atPath:altered.appendingPathComponent("continuation-parent").path))
        try originalMetrics.write(to:snapshot.directory.appendingPathComponent("metrics.json"))
        let parent = try workspace.checkpointParent(at: snapshot.directory)
        XCTAssertEqual(parent.manifestSHA256, snapshot.manifestSHA256)
        let record=try snapshot.parentRecord()
        XCTAssertEqual(record.evidence.workerRunID,parent.runID)
        XCTAssertEqual(record.evidence.checkpointSHA256,parent.checkpointSHA256)
        XCTAssertEqual(record.evidence.completedEpoch,3)
        XCTAssertEqual(record.device,"cpu")
        XCTAssertEqual(record.metrics.map(\.epoch),[1,2,3])
        var parentContents=try Dictionary(uniqueKeysWithValues:snapshot.files.map {($0.path,try Data(contentsOf:snapshot.directory.appendingPathComponent($0.path)))})
        parentContents["metrics.json"]=Data("[]".utf8)
        XCTAssertThrowsError(try WorkflowLifecycleParentRecord.decode(parentContents))
        var parityDirectories:[String:URL]=[:]
        for mode in [WorkflowTrainingMode.exactResume,.fineTune,.fresh] {
            var source=try WorkflowCodec.migrateToVersion3(outcome.request.document)
            let trainIndex=try XCTUnwrap(source.nodes.firstIndex {$0.address == WorkflowExamples.trainAddress})
            guard case .train(var trainConfig)=source.nodes[trainIndex].payload else {return XCTFail("Missing Train")}
            trainConfig["trainingMode"] = .string(mode.rawValue)
            var config=try XCTUnwrap(WorkflowTrainRunSetup.configuration(trainConfig,schemaVersion:source.schemaVersion))
            config["epochs"] = .integer(mode == .fineTune ? 2:5)
            source.nodes[trainIndex].payload = .train(config:try WorkflowTrainRunSetup.setting(config,in:trainConfig,schemaVersion:source.schemaVersion))
            let library=ProjectLibrary(directory:root.appendingPathComponent("projects-"+mode.rawValue))
            _ = try library.saveSource(.workflow(source),id:source.projectID)
            let owner=try WorkflowAuthoringStore(library:library,projectID:source.projectID,modelRegistry:host.owner.modelRegistry,dataRegistry:host.owner.dataRegistry)
            let request=try WorkflowExecutionRequest.capture(from:owner,train:WorkflowExamples.trainAddress,runtime:try XCTUnwrap(workspace.runtime),checkpointParent:mode == .fresh ? nil:selected,moduleCompositionEvidence:workspace.moduleCompositionDiscoveryEvidence)
            if mode == .exactResume {
                let cancelled=expectation(description:"Cancel before checkpoint capture starts a worker")
                var launches=0,exitCallback:((Int32,String)->Void)?,cancelledStatus:WorkflowExecutionPhase?
                let cancelledDirectory=root.appendingPathComponent(UUID().uuidString)
                let cancelledSession=WorkflowExecutionSession(request:request,owner:owner,sources:outcome.sources,directory:cancelledDirectory,
                    reservation:.init(acquire:{UUID()},release:{_ in}),checkpointSnapshot:snapshot,
                    launch:{_,_,_,_,exit in launches+=1;exitCallback=exit;return .init(cancel:{})},
                    onTerminal:{cancelledStatus=$0.status;cancelled.fulfill()})
                XCTAssertTrue(cancelledSession.start());cancelledSession.cancel();exitCallback?(1,"")
                await fulfillment(of:[cancelled],timeout:10)
                XCTAssertEqual(launches,0)
                XCTAssertEqual(cancelledStatus,.cancelled)
                XCTAssertFalse(FileManager.default.fileExists(atPath:cancelledDirectory.appendingPathComponent("capture.json").path))
                XCTAssertFalse(owner.isBusy)
            }
            let childDirectory=root.appendingPathComponent("workflow-runs/"+source.projectID+"/"+UUID().uuidString)
            let terminal=expectation(description:"Universal \(mode) completes")
            var result:WorkflowExecutionOutcome?
            let session=WorkflowExecutionSession(request:request,owner:owner,sources:outcome.sources,directory:childDirectory,
                reservation:.init(acquire:{UUID()},release:{_ in}),checkpointSnapshot:mode == .fresh ? nil:snapshot,onTerminal:{result=$0;terminal.fulfill()})
            XCTAssertTrue(session.start(),session.statusMessage)
            await fulfillment(of:[terminal],timeout:30)
            XCTAssertEqual(result?.status,.completed,result?.error ?? "Missing result")
            XCTAssertEqual(session.metrics.count,mode == .fineTune ? 2:5)
            if mode == .exactResume {
                XCTAssertEqual(Array(session.metrics.prefix(3).map(\.trainLoss)),record.metrics.map(\.trainLoss))
                XCTAssertEqual(Array(session.metrics.prefix(3).map(\.validationLoss)),record.metrics.map(\.validationLoss))
            }
            XCTAssertFalse(owner.isBusy)
            XCTAssertNil(result?.recordError)
            let childRef=WorkflowHistoryReference(projectID:source.projectID,directoryID:try XCTUnwrap(UUID(uuidString:childDirectory.lastPathComponent)))
            guard case .verified(let verified)=WorkflowHistoryReader.load(childRef,root:root) else {return XCTFail("Continuation archive must verify")}
            parityDirectories[mode.rawValue]=childDirectory.appendingPathComponent("training/artifacts")
            XCTAssertEqual(verified.plan.trainingMode,mode)
            XCTAssertEqual(verified.modelContent?.metrics.count,mode == .fineTune ? 2:5)
        }
        let parityDone=expectation(description:"Universal resume matches uninterrupted CPU state")
        var parityCode:Int32?,parityError=""
        let parityScript="""
        import json,sys,torch
        from pathlib import Path
        def equal(a,b):
            if isinstance(a,torch.Tensor): assert torch.equal(a,b)
            elif isinstance(a,dict):
                assert a.keys()==b.keys()
                for k in a: equal(a[k],b[k])
            elif isinstance(a,(list,tuple)):
                assert len(a)==len(b)
                for x,y in zip(a,b): equal(x,y)
            else: assert a==b
        full,resumed=map(Path,sys.argv[1:])
        a=torch.load(full/'checkpoint.pt',map_location='cpu',weights_only=True)
        b=torch.load(resumed/'checkpoint.pt',map_location='cpu',weights_only=True)
        for key in ('modelState','optimizerState','rngState'): equal(a[key],b[key])
        assert json.loads((full/'metrics.json').read_text())==json.loads((resumed/'metrics.json').read_text())
        """
        let parityHandle=try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:development).appendingPathComponent(".venv/bin/python"),
            ["-c",parityScript,try XCTUnwrap(parityDirectories["fresh"]).path,try XCTUnwrap(parityDirectories["exactResume"]).path],
            URL(fileURLWithPath:development),{_ in},{code,stderr in parityCode=code;parityError=stderr;parityDone.fulfill()})
        defer {parityHandle.cancel()}
        await fulfillment(of:[parityDone],timeout:30)
        XCTAssertEqual(parityCode,0,parityError)
        let hostParent=await host.selectCheckpointParent(reference:ref,for:WorkflowExamples.trainAddress,lease:host.owner.lease())
        XCTAssertTrue(hostParent)
        XCTAssertTrue(host.owner.setTrainingMode(.fineTune,for:WorkflowExamples.trainAddress,lease:host.owner.lease()))
        let beforeContinuationCount=host.outcomes.count
        let originalLauncher = host.executionLaunch
        var cancelledCaptureLaunches = 0
        host.executionLaunch = { executable, arguments, directory, line, exit in
            cancelledCaptureLaunches += 1
            return try originalLauncher(executable, arguments, directory, line, exit)
        }
        let captureStopped = expectation(description: "Cancelled checkpoint capture acknowledges staging release")
        let cancellationObservation = workspace.$isPreparingData.dropFirst().filter { !$0 }.prefix(1).sink { _ in captureStopped.fulfill() }
        XCTAssertTrue(host.run(train:WorkflowExamples.trainAddress))
        XCTAssertTrue(host.isStagingCheckpoint)
        XCTAssertTrue(workspace.isBusy)
        XCTAssertFalse(host.run(train:WorkflowExamples.trainAddress),"Duplicate Run must not stage another checkpoint")
        workspace.cancelRun()
        // Capture owns its reservation until the asynchronous task has stopped.
        await fulfillment(of: [captureStopped], timeout: 10)
        cancellationObservation.cancel()
        XCTAssertEqual(cancelledCaptureLaunches, 0)
        host.executionLaunch = originalLauncher
        XCTAssertFalse(host.isStagingCheckpoint)
        XCTAssertFalse(workspace.isBusy)
        XCTAssertEqual(host.outcomes.count,beforeContinuationCount)
        let hostDone=expectation(description:"Workspace Run uses selected checkpoint")
        let hostObservation=host.$outcomes.dropFirst().prefix(1).sink {_ in hostDone.fulfill()}
        XCTAssertTrue(host.run(train:WorkflowExamples.trainAddress),host.statusMessage)
        await fulfillment(of:[hostDone],timeout:30);hostObservation.cancel()
        XCTAssertEqual(host.outcomes.count,beforeContinuationCount+1)
        XCTAssertFalse(host.isStagingCheckpoint)
        let hostResult=try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(hostResult.status,.completed,hostResult.error ?? "")
        XCTAssertEqual(hostResult.request.plan.trainingMode,.fineTune)
        XCTAssertEqual(hostResult.request.checkpointParent,selected)
        let checkpoint = outcome.directory.appendingPathComponent("training/artifacts/checkpoint.pt")
        try Data("changed".utf8).write(to: checkpoint)
        XCTAssertThrowsError(try WorkflowHistoryReader.stageCheckpoint(ref, root: root))
        try FileManager.default.removeItem(at: outcome.directory)
        XCTAssertEqual(try workspace.checkpointParent(at: snapshot.directory), parent)
        let graph = root.appendingPathComponent("finetune-graph.json")
        try JSONEncoder().encode(parent.graph).write(to: graph)
        let runtime = try XCTUnwrap(workspace.runtime), output = root.appendingPathComponent("finetuned")
        let finished = expectation(description: "Fine-tune from private snapshot")
        var exitCode: Int32?, errorText = ""
        let handle = try WorkflowExecutionSession.launchWorker(runtime.executableURL, runtime.argumentPrefix + [
            "--graph", graph.path, "--output", output.path, "--epochs", "2", "--device", "cpu",
            "--finetune-checkpoint", snapshot.directory.appendingPathComponent("checkpoint.pt").path], runtime.workingDirectoryURL,
            { _ in }, { code, stderr in exitCode = code; errorText = stderr; finished.fulfill() })
        defer { handle.cancel() }
        await fulfillment(of: [finished], timeout: 30)
        XCTAssertEqual(exitCode, 0, errorText)
        let trained = try workspace.checkpointParent(at: output)
        XCTAssertEqual(trained.parentRunID, parent.runID)
        XCTAssertNotEqual(trained.runID, parent.runID)
        XCTAssertEqual(trained.completedEpoch, 2)
        let lifecycle=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:output.appendingPathComponent("manifest.json")))
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validate(lifecycle,mode:.fineTune,targetEpochs:2,
            parent:.init(workerRunID:parent.runID,checkpointSHA256:parent.checkpointSHA256,completedEpoch:parent.completedEpoch)))
        XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(lifecycle,mode:.exactResume,targetEpochs:2,
            parent:.init(workerRunID:parent.runID,checkpointSHA256:parent.checkpointSHA256,completedEpoch:parent.completedEpoch)))
        XCTAssertEqual(try workspace.checkpointParent(at: snapshot.directory), parent)
    }
}
