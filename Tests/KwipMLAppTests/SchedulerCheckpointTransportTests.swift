import XCTest
import Darwin
import KwipMLCore
@testable import KwipMLApp

@MainActor final class SchedulerCheckpointTransportTests:XCTestCase {
    func testActualObjectiveScheduledParentTraceSurvivesArchiveAndExactCPUResume()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Enable actual owned worker")}
        guard let canonical=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        defer {free(canonical)}
        let root=URL(fileURLWithPath:String(cString:canonical)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let repo=URL(fileURLWithPath:development)
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        let runtime=try XCTUnwrap(workspace.runtime)
        let compositionEvidence=try XCTUnwrap(workspace.moduleCompositionDiscoveryEvidence)
        let registry=try ComponentRegistry.builtIn(),data=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        var document=try WorkflowExamples.preparedLossObjective(task:.mse,device:"cpu")
        let template=try WorkflowExamples.preparedStepLRClassifier(device:"cpu")
        let templateTrain=try XCTUnwrap(template.nodes.first {$0.address == WorkflowExamples.trainAddress})
        let train=try XCTUnwrap(document.nodes.firstIndex {$0.address == WorkflowExamples.trainAddress})
        guard case .train(let supplied)=templateTrain.payload,case .train(var trainSettings)=document.nodes[train].payload else {return XCTFail("Missing Train controls")}
        for role in ["optimizer","scheduler"] {trainSettings[role]=try XCTUnwrap(supplied[role])}
        document.nodes[train].payload = .train(config:trainSettings)
        func configured(_ epochs:Int,_ mode:WorkflowTrainingMode)->WorkflowDocument {
            var value=document
            for index in value.nodes.indices {
                if case .runSetup(var config)=value.nodes[index].payload {config["epochs"] = .integer(Int64(epochs));value.nodes[index].payload = .runSetup(config:config)}
                if case .train(var config) = value.nodes[index].payload {
                    config["trainingMode"] = .string(mode.rawValue)
                    if mode == .fineTune {config.removeValue(forKey:"scheduler")}
                    value.nodes[index].payload = .train(config:config)
                }
            }
            return value
        }
        let source=try WorkflowCapturedSource.capture(repo.appendingPathComponent("Sources/KwipMLApp/Resources/loss-mse-observations.csv"))
        func execute(_ value:WorkflowDocument,_ label:String,_ snapshot:WorkflowCheckpointSnapshot?=nil,_ parent:WorkflowCheckpointReference?=nil)async throws->(WorkflowExecutionOutcome,WorkflowExecutionSession) {
            let library=ProjectLibrary(directory:root.appendingPathComponent("projects-"+label));_ = try library.saveSource(.workflow(value),id:value.projectID)
            let owner=try WorkflowAuthoringStore(library:library,projectID:value.projectID,modelRegistry:registry,dataRegistry:data)
            let request=try WorkflowExecutionRequest.capture(from:owner,train:WorkflowExamples.trainAddress,runtime:runtime,checkpointParent:parent,moduleCompositionEvidence:compositionEvidence)
            XCTAssertEqual(request.plan.lossRecipe?.configuration.componentID,"loss_mse")
            XCTAssertNotNil(request.plan.optimizerRecipe)
            let done=expectation(description:label);var result:WorkflowExecutionOutcome?
            let directory=root.appendingPathComponent("workflow-runs/\(value.projectID)/\(UUID().uuidString)")
            let session=WorkflowExecutionSession(request:request,owner:owner,sources:[WorkflowExamples.sourceAddress:source],directory:directory,reservation:.init(acquire:{UUID()},release:{_ in}),checkpointSnapshot:snapshot,onTerminal:{result=$0;done.fulfill()})
            defer {if owner.isBusy {session.cancel()};owner.invalidate()}
            XCTAssertTrue(session.start(),session.statusMessage)
            await fulfillment(of:[done],timeout:60)
            let outcome=try XCTUnwrap(result)
            XCTAssertEqual(outcome.status,.completed,outcome.error ?? session.statusMessage)
            XCTAssertNil(outcome.recordError)
            guard outcome.status == .completed else {throw WorkflowPhaseError("Actual scheduled objective execution failed")}
            return(outcome,session)
        }
        let (parent,_)=try await execute(configured(2,.fresh),"parent")
        let ref=WorkflowHistoryReference(projectID:document.projectID,directoryID:try XCTUnwrap(UUID(uuidString:parent.directory.lastPathComponent)))
        guard case .verified=WorkflowHistoryReader.load(ref,root:root) else {return XCTFail("Actual parent History must verify")}
        let selected=WorkflowCheckpointReference(projectID:ref.projectID,directoryID:ref.directoryID.uuidString,runID:parent.runID.uuidString,requestIdentity:parent.request.requestIdentity,modelManifestSHA256:try XCTUnwrap(parent.modelManifestSHA256))
        let snapshot=try WorkflowHistoryReader.stageCheckpoint(selected,root:root)
        defer {snapshot.remove()}
        XCTAssertEqual(Set(snapshot.files.map(\.path)),["manifest.json","checkpoint.pt","metrics.json","rate-trace.json"])
        let traceURL=snapshot.directory.appendingPathComponent("rate-trace.json")
        let trace=try Data(contentsOf:traceURL)
        let archived=root.appendingPathComponent("archive-probe");try FileManager.default.createDirectory(at:archived,withIntermediateDirectories:true)
        let files=try snapshot.archive(to:archived)
        XCTAssertTrue(files.contains {$0.path == "continuation-parent/rate-trace.json" && $0.sha256 == WorkflowArchiveCodec.hash(trace)})
        XCTAssertTrue(try snapshot.parentRecord().evidence.hasScheduler)
        for missing in [false,true] {
            let destination=root.appendingPathComponent("refuse-\(missing)");try FileManager.default.createDirectory(at:destination,withIntermediateDirectories:true)
            if missing {try FileManager.default.removeItem(at:traceURL)} else {try Data("changed".utf8).write(to:traceURL)}
            XCTAssertThrowsError(try snapshot.archive(to:destination))
            XCTAssertFalse(FileManager.default.fileExists(atPath:destination.appendingPathComponent("continuation-parent").path))
            try trace.write(to:traceURL)
        }
        let (resumed,resumedSession)=try await execute(configured(4,.exactResume),"resumed",snapshot,selected)
        let fineSnapshot=try WorkflowHistoryReader.stageCheckpoint(selected,root:root)
        defer {fineSnapshot.remove()}
        let (fine,fineSession)=try await execute(configured(2,.fineTune),"fine-without-child-schedule",fineSnapshot,selected)
        XCTAssertEqual(fineSession.metrics.map(\.epoch),[1,2])
        XCTAssertNil(fine.request.plan.schedulerRecipe)
        XCTAssertFalse(FileManager.default.fileExists(atPath:fine.directory.appendingPathComponent("training/artifacts/rate-trace.json").path))
        let fineCapture=try JSONSerialization.jsonObject(with:Data(contentsOf:fine.directory.appendingPathComponent("capture.json"))) as? [String:Any]
        let fineFiles=fineCapture?["checkpointFiles"] as? [[String:Any]]
        XCTAssertTrue(fineFiles?.contains {$0["path"] as? String == "continuation-parent/rate-trace.json"} == true)
        let (full,fullSession)=try await execute(configured(4,.fresh),"full")
        XCTAssertEqual(resumedSession.metrics.map(\.trainLoss),fullSession.metrics.map(\.trainLoss))
        XCTAssertEqual(resumedSession.metrics.map(\.validationLoss),fullSession.metrics.map(\.validationLoss))
        XCTAssertTrue(resumedSession.metrics.allSatisfy {$0.accuracy == nil})
        let parity=expectation(description:"Actual checkpoint tensors and schedule parity");var code:Int32?,error=""
        let script="""
        import sys,json,torch
        from pathlib import Path
        a,b=map(Path,sys.argv[1:])
        def equal(x,y):
            if isinstance(x,torch.Tensor): assert torch.equal(x,y)
            elif isinstance(x,dict):
                assert x.keys()==y.keys()
                for k in x: equal(x[k],y[k])
            elif isinstance(x,(list,tuple)):
                assert len(x)==len(y)
                for u,v in zip(x,y): equal(u,v)
            else: assert x==y
        x=torch.load(a/'checkpoint.pt',weights_only=True,map_location='cpu');y=torch.load(b/'checkpoint.pt',weights_only=True,map_location='cpu')
        for k in ('modelState','optimizerState','optimizerUpdates','rngState','objectiveState','scheduler'): equal(x[k],y[k])
        for name in ('metrics.json','rate-trace.json'): assert json.loads((a/name).read_text())==json.loads((b/name).read_text())
        """
        let handle=try WorkflowExecutionSession.launchWorker(repo.appendingPathComponent(".venv/bin/python"),["-c",script,full.directory.appendingPathComponent("training/artifacts").path,resumed.directory.appendingPathComponent("training/artifacts").path],repo,{_ in},{status,message in code=status;error=message;parity.fulfill()})
        defer {handle.cancel()}
        await fulfillment(of:[parity],timeout:30)
        XCTAssertEqual(code,0,error)
        guard case .verified=WorkflowHistoryReader.load(ref,root:root) else {return XCTFail("Original parent remains independently verified after all child runs")}
    }
}
