import XCTest
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class WorkflowExperimentRecoveryEvidenceTests:XCTestCase {
    @MainActor func testActualUnpinnedTerminalOutcomeRequiresExactCapturedMembership()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Set development root for actual recovery evidence worker fixture.")}
        let canonical=try XCTUnwrap(realpath(FileManager.default.temporaryDirectory.path,nil));defer {free(canonical)}
        let root=URL(fileURLWithPath:String(cString:canonical)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
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
        let done=expectation(description:"actual experiment outcomes")
        let runner=try WorkflowExperimentRunner(requests:requests,owner:host.owner,group:group,groupOwner:owned.owner,workspaceDirectory:root,reservation:.init(acquire:{UUID()},release:{_ in}),onTerminal:{done.fulfill()})
        XCTAssertTrue(runner.start());await fulfillment(of:[done],timeout:90)
        guard runner.outcomes.count == 2,runner.outcomes.allSatisfy({$0.status == .completed}) else {return XCTFail(runner.error ?? "Actual fixture did not complete")}
        let ledger=owned.directory.appendingPathComponent("group.json"),original=try Data(contentsOf:ledger)
        XCTAssertEqual(try WorkflowExperimentRecoveryCoordinator.recover(directory:owned.directory,workspaceDirectory:root),group.record,"An already completed group needs no recovery transition.")
        XCTAssertEqual(try Data(contentsOf:ledger),original,"Recovery preserves a completed ledger byte-for-byte.")
        var interrupted=group.record
        interrupted.variants[0].status = .running;interrupted.variants[0].outcomeSHA256=nil
        try WorkflowArchiveCodec.encode(interrupted).write(to:ledger)
        let context=try WorkflowExperimentHistoryReader.captureContext(directory:owned.directory)
        let verified=try WorkflowExperimentHistoryReader.verifiedVariant(context:context,ordinal:1,workspaceDirectory:root,policy:.recoveryRunning)
        XCTAssertEqual(verified.run.outcome.runID,runner.outcomes[0].runID.uuidString)
        XCTAssertEqual(verified.outcomeSHA256,WorkflowPhaseArtifacts.hash(try Data(contentsOf:runner.outcomes[0].directory.appendingPathComponent("outcome.json"))))
        XCTAssertThrowsError(try WorkflowExperimentHistoryReader.verifiedVariant(context:context,ordinal:1,workspaceDirectory:root,policy:.historyPinned))
        XCTAssertThrowsError(try WorkflowExperimentHistoryReader.verifiedVariant(context:context,ordinal:2,workspaceDirectory:root,policy:.recoveryRunning))
        let history=try WorkflowExperimentHistoryReader.load(directory:owned.directory,workspaceDirectory:root)
        if case .recorded(.running)=history.variants[0].evidence {} else {XCTFail("Normal History must not promote unpinned running outcome")}
        var wrong=interrupted
        wrong.variants[0].runID=wrong.variants[1].runID;wrong.variants[0].directoryID=wrong.variants[1].directoryID
        try WorkflowArchiveCodec.encode(wrong).write(to:ledger)
        let substituted=try WorkflowExperimentHistoryReader.captureContext(directory:owned.directory)
        XCTAssertThrowsError(try WorkflowExperimentHistoryReader.verifiedVariant(context:substituted,ordinal:1,workspaceDirectory:root,policy:.recoveryRunning))
        wrong=interrupted;wrong.variants[0] = .init(ordinal:1,requestIdentity:String(repeating:"b",count:64),status:.running,runID:interrupted.variants[0].runID,directoryID:interrupted.variants[0].directoryID,outcomeSHA256:nil)
        try WorkflowArchiveCodec.encode(wrong).write(to:ledger)
        XCTAssertThrowsError(try WorkflowExperimentHistoryReader.captureContext(directory:owned.directory))
        try original.write(to:ledger)
        let restored=try WorkflowExperimentHistoryReader.captureContext(directory:owned.directory)
        XCTAssertNoThrow(try WorkflowExperimentHistoryReader.verifiedVariant(context:restored,ordinal:1,workspaceDirectory:root,policy:.historyPinned))
        let source=try XCTUnwrap(inputs.sources.values.first).url
        let sourceBytes=try Data(contentsOf:source)
        try Data("changed source".utf8).write(to:source)
        XCTAssertThrowsError(try WorkflowExperimentHistoryReader.captureContext(directory:owned.directory))
        try sourceBytes.write(to:source)
        let outcomeFile=runner.outcomes[0].directory.appendingPathComponent("outcome.json")
        let outcomeBytes=try Data(contentsOf:outcomeFile)
        try WorkflowArchiveCodec.encode(interrupted).write(to:ledger)
        let recovered=try WorkflowExperimentRecoveryCoordinator.recover(directory:owned.directory,workspaceDirectory:root)
        XCTAssertEqual(recovered.variants[0].status,.completed)
        XCTAssertEqual(recovered.variants[0].outcomeSHA256,WorkflowPhaseArtifacts.hash(outcomeBytes))
        XCTAssertEqual(recovered.variants[1],interrupted.variants[1],"Verified terminal sibling is preserved exactly.")
        XCTAssertEqual(try Data(contentsOf:outcomeFile),outcomeBytes,"Recovery never rewrites outcome evidence.")
        let recoveredBytes=try Data(contentsOf:ledger)
        XCTAssertEqual(try WorkflowExperimentRecoveryCoordinator.recover(directory:owned.directory,workspaceDirectory:root),recovered)
        XCTAssertEqual(try Data(contentsOf:ledger),recoveredBytes,"Repeated reconciliation does not advance or rewrite the ledger.")
        // Reuse the actual archive, but independently corrupt and remove its receipt.
        for missing in [false,true] {
            try WorkflowArchiveCodec.encode(interrupted).write(to:ledger)
            if missing {try FileManager.default.removeItem(at:outcomeFile)} else {try Data("{}".utf8).write(to:outcomeFile)}
            let failed=try WorkflowExperimentRecoveryCoordinator.recover(directory:owned.directory,workspaceDirectory:root)
            XCTAssertEqual(failed.variants[0].status,.interrupted)
            XCTAssertNil(failed.variants[0].outcomeSHA256)
            XCTAssertEqual(failed.variants[1],interrupted.variants[1])
            if !missing {XCTAssertEqual(try Data(contentsOf:outcomeFile),Data("{}".utf8))}
            else {XCTAssertFalse(FileManager.default.fileExists(atPath:outcomeFile.path))}
            try outcomeBytes.write(to:outcomeFile)
        }
    }
}
