import XCTest
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class WorkflowExperimentInterphaseRecoveryTests:XCTestCase {
    private let roleKey="KWIPML_INTERPHASE_CHILD_ROOT"
    /// This barrier is after real process exit and independent preparation
    /// verification, before delivering the exit callback to the session. It does
    /// not claim to pause inside the session's private synchronous receiveExit.
    @MainActor func testVerifiedPreparationBeforeCallbackSurvivesHostLoss()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Actual preparation requires development runtime")}
        if let childRoot=ProcessInfo.processInfo.environment[roleKey] {
            try await runChild(root:URL(fileURLWithPath:childRoot),development:development)
            return
        }
        let canonical=try XCTUnwrap(realpath(FileManager.default.temporaryDirectory.path,nil));defer {free(canonical)}
        let root=URL(fileURLWithPath:String(cString:canonical)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
        let child=Process(),exitAcknowledged=DispatchSemaphore(value:0)
        var childLaunched=false,preparationExited=false
        defer {
            if childLaunched && child.isRunning {
                terminateAndAwaitFixtureExit(child,acknowledgement:exitAcknowledged)
            }
            // Before the barrier, host exit alone does not establish worker exit.
            // Preserve failed-fixture evidence instead of deleting active inputs.
            if !childLaunched || (!child.isRunning && preparationExited) {
                try? FileManager.default.removeItem(at:root)
            }
        }
        let marker=root.appendingPathComponent("verified-boundary.json"),log=root.appendingPathComponent("child.log")
        FileManager.default.createFile(atPath:log.path,contents:nil)
        let output=try FileHandle(forWritingTo:log);defer {try? output.close()}
        let exited=expectation(description:"Owned XCTest host exited")
        child.executableURL=URL(fileURLWithPath:"/usr/bin/xcrun")
        child.arguments=["xctest","-XCTest","KwipMLAppTests.WorkflowExperimentInterphaseRecoveryTests/testVerifiedPreparationBeforeCallbackSurvivesHostLoss",Bundle(for:Self.self).bundleURL.path]
        var environment=ProcessInfo.processInfo.environment;environment[roleKey]=root.path
        child.environment=environment;child.standardOutput=output;child.standardError=output
        child.terminationHandler={_ in exitAcknowledged.signal();exited.fulfill()}
        try child.run()
        childLaunched=true
        let deadline=ContinuousClock.now.advanced(by:.seconds(60))
        while !FileManager.default.fileExists(atPath:marker.path),child.isRunning,ContinuousClock.now<deadline {
            try await Task.sleep(for:.milliseconds(25))
        }
        guard FileManager.default.fileExists(atPath:marker.path) else {
            if child.isRunning {child.terminate()}
            await fulfillment(of:[exited],timeout:12)
            return XCTFail("Child did not reach verified preparation barrier: \((try? String(contentsOf:log,encoding:.utf8)) ?? "No log")")
        }
        let info=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:marker)) as? [String:String])
        preparationExited=true
        let group=URL(fileURLWithPath:try XCTUnwrap(info["group"])),run=try XCTUnwrap(info["run"])
        let runDirectory=URL(fileURLWithPath:try XCTUnwrap(info["directory"]))
        let original=try WorkflowExperimentGroupStore.readRecord(directory:group)
        XCTAssertEqual(original.variants.map(\.status),[.running,.pending])
        XCTAssertEqual(try WorkflowExperimentRecoveryCoordinator.probe(directory:group,workspaceDirectory:root),.inUse)
        let prepared=try snapshot(runDirectory.appendingPathComponent("preparation"))
        XCTAssertFalse(prepared.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath:group.appendingPathComponent("phase-\(run)-training.json").path))
        // Process retains ownership of this exact host; no cached PID is used.
        child.terminate()
        await fulfillment(of:[exited],timeout:12)
        guard !child.isRunning else {return XCTFail("Owned test host did not exit")}
        XCTAssertNotEqual(child.terminationStatus,0)
        let recovered=try WorkflowExperimentRecoveryCoordinator.recover(directory:group,workspaceDirectory:root)
        XCTAssertEqual(recovered.variants.map(\.status),[.interrupted,.cancelled])
        XCTAssertNil(recovered.variants[0].outcomeSHA256)
        XCTAssertEqual(try snapshot(runDirectory.appendingPathComponent("preparation")),prepared)
        XCTAssertFalse(FileManager.default.fileExists(atPath:runDirectory.appendingPathComponent("training").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath:runDirectory.appendingPathComponent("outcome.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath:group.appendingPathComponent("phase-\(run)-training.json").path))
        let names=try FileManager.default.contentsOfDirectory(atPath:group.path)
        XCTAssertEqual(names.filter {$0.hasPrefix("claim-")}.count,1,"Only preparation was authorized and claimed")
        XCTAssertEqual(try WorkflowExperimentRecoveryCoordinator.recover(directory:group,workspaceDirectory:root),recovered)
    }
    /// Synchronous, test-only failure cleanup; the termination handler signals
    /// from Foundation's queue, so it does not require the main actor to advance.
    private func terminateAndAwaitFixtureExit(_ child:Process,acknowledgement:DispatchSemaphore) {
        child.terminate()
        _=acknowledgement.wait(timeout:.now()+12)
    }
    private func snapshot(_ root:URL)throws->[String:Data] {
        var result:[String:Data]=[:]
        let files=try XCTUnwrap(FileManager.default.enumerator(at:root,includingPropertiesForKeys:[.isRegularFileKey]))
        for case let file as URL in files {
            if try file.resourceValues(forKeys:[.isRegularFileKey]).isRegularFile == true {result[file.path]=try Data(contentsOf:file)}
        }
        return result
    }
    @MainActor private func runChild(root:URL,development:String)async throws {
        // A lost test parent cannot strand this test host indefinitely. No product hook.
        DispatchQueue.global().asyncAfter(deadline:.now()+75) {Darwin._exit(124)}
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        var document=try WorkflowCodec.migrateToVersion4(WorkflowExamples.preparedParallelClassifier(device:"cpu"))
        let index=try XCTUnwrap(document.nodes.firstIndex {if case .runSetup=$0.payload {return true};return false})
        guard case .runSetup(var config)=document.nodes[index].payload else {return XCTFail("Missing setup")}
        config["experiment"] = .object(["schemaVersion":1,"axes":.array([.object(["field":.string("seed"),"values":.array([7,8])])]),"budget":.object(["maximumRuns":2,"maximumTotalEpochs":6])])
        document.nodes[index].payload = .runSetup(config:config)
        _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID))
        let host=try XCTUnwrap(workspace.workflowSession),runtime=try XCTUnwrap(workspace.runtime)
        let base=try WorkflowExecutionRequest.capture(from:host.owner,train:WorkflowExamples.trainAddress,runtime:runtime,moduleCompositionEvidence:workspace.moduleCompositionDiscoveryEvidence)
        let owned=try WorkflowGroupOwnership.createGroup(workspace:root,projectID:host.owner.projectID)
        let csv=URL(fileURLWithPath:development).appendingPathComponent("Sources/KwipMLApp/Resources/workflow-observations.csv")
        let inputs=try WorkflowExperimentInputs.captureBytes(addresses:base.plan.sourceAddresses,directory:owned.directory) {_ in try Data(contentsOf:csv)}
        let requests=try base.experimentVariants(inputs:inputs,owner:host.owner)
        let group=try WorkflowExperimentGroupStore.create(requests:requests,directory:owned.directory,groupOwner:owned.owner)
        var completion:[String:Any]?,launches=0
        let launcher:WorkflowWorkerLauncher = { executable,arguments,directory,line,_ in
            launches+=1
            guard launches == 1 else {throw DataFlowContractError("Training must not launch at this boundary")}
            return try WorkflowExecutionSession.launchWorker(executable,arguments,directory,{event in
                if let data=event.data(using:.utf8),let object=(try? JSONSerialization.jsonObject(with:data)) as? [String:Any],object["event"] as? String == "dataflowCompleted" {completion=object}
                line(event)
            },{code,stderr in
                do {
                    guard code == 0 else {throw DataFlowContractError("Preparation failed: \(stderr)")}
                    let variant=group.record.variants[0]
                    let runDirectory=root.appendingPathComponent("workflow-runs/\(group.record.projectID)/\(variant.directoryID!)")
                    let verified=try WorkflowPhaseArtifacts.preparation(request:requests[0],root:runDirectory,completion:try XCTUnwrap(completion))
                    let bytes=try JSONSerialization.data(withJSONObject:["group":owned.directory.path,"run":variant.runID!,"directory":runDirectory.path,"manifest":verified.selection.manifestSHA256])
                    try bytes.write(to:root.appendingPathComponent("verified-boundary.json"),options:.atomic)
                    // Intentionally retain the host without forwarding actual exit.
                } catch {XCTFail("Boundary verification failed: \(error)");Darwin._exit(123)}
            })
        }
        let runner=try WorkflowExperimentRunner(requests:requests,owner:host.owner,group:group,groupOwner:owned.owner,workspaceDirectory:root,reservation:.init(acquire:{UUID()},release:{_ in}),launch:launcher,onTerminal:{XCTFail("No terminal callback belongs before host loss")})
        XCTAssertTrue(runner.start())
        while true {try await Task.sleep(for:.seconds(1))}
    }
}
