import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class TextGenerationIntegrationTests:XCTestCase {
    private func wait(_ predicate:()->Bool)async throws {
        for _ in 0..<1500 {if predicate() {return};try await Task.sleep(nanoseconds:20_000_000)}
        throw WorkflowArchiveError("Timed out waiting for the real generation integration.")
    }
    func testRealModelGeneratesWithCaptureOffKeepsWeightsAndVerifiedReplay()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Requires actual local runtime")}
        let root=try canonicalBackendTemporaryDirectory()
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(TextGenerationPracticeExample.exampleID,device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession),owner=host.owner
        defer {host.cancel();if !workspace.isBusy {try? FileManager.default.removeItem(at:root)}}
        try owner.setRunHistoryPolicy(.init(metricsAndDiagnostics:false,componentOutputs:false))
        XCTAssertTrue(host.run(train:WorkflowExamples.trainAddress),host.statusMessage)
        try await wait {(host.frozenBindings.reference(for:WorkflowExamples.trainAddress) != nil || host.statusMessage.hasPrefix("Trained model unavailable")) && !workspace.isBusy && !owner.isBusy}
        let reference=try XCTUnwrap(host.frozenBindings.reference(for:WorkflowExamples.trainAddress),host.statusMessage),count=host.outcomes.count
        let metadata=try await host.generationModelMetadata(for:TextGenerationPracticeExample.address)
        XCTAssertEqual(metadata.context,3);XCTAssertEqual(metadata.vocabularySize,4);XCTAssertFalse(metadata.hasEOS);XCTAssertTrue(metadata.hasUnknown)
        // Metadata verification yields while completed-model publication and
        // captured-output reads can still acquire their operation reservation.
        // Invoke the manual action only after its documented idle precondition.
        try await wait {!workspace.isBusy && !owner.isBusy && !host.isManagingTrainedArtifacts && host.generation?.isBusy != true}
        XCTAssertTrue(host.generateText(address:TextGenerationPracticeExample.address),host.statusMessage)
        try await wait {!workspace.isBusy && !host.isManagingTrainedArtifacts && host.generation?.isBusy != true}
        let generation=try XCTUnwrap(host.generation,host.statusMessage),record=try XCTUnwrap(generation.record,generation.statusMessage)
        XCTAssertEqual(record.status,"completed",generation.statusMessage);XCTAssertNotNil(generation.artifact)
        XCTAssertEqual(host.outcomes.count,count,"Generate must not train behind the action")
        let verified=try record.verify(root:root)
        guard case .array(let rows)=verified.result["rows"],case .object(let first)=rows.first else {return XCTFail("Missing complete generated row")}
        XCTAssertEqual(first["generated_text"],.string("babababababababa"));XCTAssertEqual(first["new_tokens"],.integer(16))
        let checkpoint=try Data(contentsOf:record.directory(root:root).appendingPathComponent("source/checkpoint.pt"))
        XCTAssertEqual(WorkflowArchiveCodec.hash(checkpoint),reference.checkpointSHA256)
        let export=root.appendingPathComponent("generation-replay"),pending=try WorkflowGenerationExport.stage(record,root:root,to:export)
        defer {pending.close()};try pending.commit()
        XCTAssertTrue(FileManager.default.fileExists(atPath:export.appendingPathComponent("generate.py").path))
        let records=try WorkflowGenerationRecord.list(root:root,projectID:owner.projectID)
        XCTAssertTrue(records.contains{$0.id==record.id})
        XCTAssertTrue(workspace.beginWorkflowProject(id:owner.projectID,forceReload:true),workspace.statusMessage)
        let restored=try XCTUnwrap(workspace.workflowSession)
        try await wait {restored.generation?.record?.id==record.id || restored.statusMessage.hasPrefix("Captured generation could not")}
        XCTAssertEqual(restored.generation?.record?.id,record.id,restored.statusMessage)
        let view=WorkflowNodeAddress(domain:.data,localID:"generation-view")
        XCTAssertTrue(restored.canvas.apply(.insert(.init(address:view,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))),lease:restored.owner.lease()))
        XCTAssertTrue(restored.canvas.apply(.connect(from:.init(address:record.address,port:"table"),to:.init(address:view,port:"table")),lease:restored.owner.lease()))
        XCTAssertTrue(restored.owner.select(.init(nodeAddresses:[view]),lease:restored.owner.lease()))
        XCTAssertTrue(restored.run(),restored.statusMessage)
        try await wait {restored.predictionPreparation?.isRunning == false && !workspace.isBusy}
        XCTAssertEqual(restored.predictionPreparation?.phase,.completed,restored.statusMessage)
        XCTAssertNotNil(restored.capturedTableViewSession(for:.init(address:view,port:"table")),"The canvas must expose the completed View, not only its archive receipt")
        let corrupted=record.directory(root:root).appendingPathComponent("output/generation.json")
        try Data("{}".utf8).write(to:corrupted)
        XCTAssertThrowsError(try record.verify(root:root),"History must reject changed output authority")
    }
    func testWholeFlowWithoutSelectionGeneratesPublishesViewAndAdvancesModelConsumers()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Requires actual local runtime")}
        let root=try canonicalBackendTemporaryDirectory()
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(TextGenerationPracticeExample.exampleID,device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession),owner=host.owner
        defer {host.cancel();if !workspace.isBusy {try? FileManager.default.removeItem(at:root)}}
        let view=WorkflowNodeAddress(domain:.data,localID:"generation-view"),save=WorkflowNodeAddress(domain:.control,localID:"z-save")
        XCTAssertTrue(host.canvas.apply(.insert(.init(address:view,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))),lease:owner.lease()))
        XCTAssertTrue(host.canvas.apply(.connect(from:.init(address:TextGenerationPracticeExample.address,port:"table"),to:.init(address:view,port:"table")),lease:owner.lease()))
        XCTAssertTrue(host.canvas.apply(.insert(.init(address:save,payload:.saveModel(config:[:]))),lease:owner.lease()))
        XCTAssertTrue(host.canvas.apply(.connect(from:.init(address:WorkflowExamples.trainAddress,port:"model"),to:.init(address:save,port:"model")),lease:owner.lease()))
        XCTAssertTrue(host.modelDestinations.set(destination:root.appendingPathComponent("model.pt"),for:save,lease:owner.lease()))
        XCTAssertTrue(owner.select(.init(),lease:owner.lease()))
        XCTAssertTrue(host.run(),host.statusMessage)
        try await wait {host.modelSaveReceiptsByAddress[save] != nil && !workspace.isBusy && !owner.isBusy && !host.isManagingTrainedArtifacts}
        XCTAssertEqual(host.outcomes.count,1,"The whole flow trains once for all connected model consumers")
        guard let generation=host.generation else {return XCTFail("Whole-flow Run skipped the unselected Generate Text consumer")}
        XCTAssertEqual(generation.record?.status,"completed",generation.statusMessage)
        XCTAssertNotNil(host.snapshot(for:TextGenerationPracticeExample.address)?.ports["table"],"Completed output must be published without selecting Generate Text")
        try await wait {host.predictionPreparation?.phase == .completed && !workspace.isBusy && !owner.isBusy}
        XCTAssertNotNil(host.snapshot(for:view),"Connected View must publish independently of selection")
        XCTAssertTrue(owner.selection.nodeAddresses.isEmpty,"Running a flow must not require or change component selection")
        XCTAssertTrue(owner.select(.init(nodeAddresses:[view]),lease:owner.lease()))
        XCTAssertNotNil(host.capturedTableViewSession(for:.init(address:view,port:"table")),"Selecting the completed View must expose its verified rows")
    }
}
