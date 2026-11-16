import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class MixtureExpertsIntegrationTests:XCTestCase {
    private func wait(_ detail:@autoclosure ()->String = "",_ predicate:()->Bool)async throws {
        let limit=Date().addingTimeInterval(65)
        while Date()<limit {if predicate(){return};try await Task.sleep(for:.milliseconds(20))}
        throw WorkflowArchiveError("Timed out awaiting real MoE execution/query: \(detail())")
    }
    func testRealTrainRoutingViewHistoryAndColdTableQuery()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Needs explicit local runtime")}
        _=NSApplication.shared
        let root=try canonicalBackendTemporaryDirectory(),workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(MixtureExpertsPractice.lessonID,device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        defer{host.cancel();if !workspace.isBusy{try? FileManager.default.removeItem(at:root)}}
        // Exercise the new inline authoring authority before the actual worker captures source.
        let member=WorkflowInlineContainer.address(root:.init(domain:.model,localID:"mixture"),childID:"expert_a/hidden")
        XCTAssertTrue(host.owner.select(.init(nodeAddresses:[member]),lease:host.owner.lease()))
        let layer=try WorkflowModelNodeContext(owner:host.owner,address:member)
        XCTAssertTrue(layer.setField("bias",value:false,expected:layer.configuration),layer.statusMessage)
        XCTAssertTrue(host.owner.select(.init(nodeAddresses:[WorkflowExamples.trainAddress]),lease:host.owner.lease()))
        let context=try WorkflowControlNodeContext(owner:host.owner,address:WorkflowExamples.trainAddress)
        var config=context.configuration
        if case .object(var settings)=config["runSetup"]{settings["epochs"]=3;settings["batchSize"]=32;config["runSetup"] = .object(settings)}
        XCTAssertTrue(context.authority.configure(config,expected:context.configuration),context.statusMessage)
        let view=WorkflowNodeAddress(domain:.data,localID:"routing-view")
        XCTAssertTrue(host.canvas.apply(.insert(.init(address:view,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))),lease:host.owner.lease()))
        XCTAssertTrue(host.canvas.apply(.connect(from:.init(address:WorkflowExamples.trainAddress,port:"routing"),to:.init(address:view,port:"table")),lease:host.owner.lease()))
        XCTAssertTrue(host.owner.select(.init(),lease:host.owner.lease()))
        XCTAssertTrue(host.run(),host.statusMessage)
        try await wait("root=\(root.path) active=\(host.hasActiveOperation) busy=\(workspace.isBusy) host=\(host.statusMessage) execution=\(String(describing:host.currentExecution?.phase)) \(host.currentExecution?.statusMessage ?? "none") prep=\(String(describing:host.predictionPreparation?.phase))"){!host.hasActiveOperation && !workspace.isBusy}
        let execution=try XCTUnwrap(host.currentExecution,host.statusMessage)
        XCTAssertEqual(execution.phase,.completed,execution.statusMessage)
        let evidence=try XCTUnwrap(execution.routingTables["routing"],execution.statusMessage)
        XCTAssertEqual(evidence.rowCount,8)
        XCTAssertTrue(execution.liveTrainingProgress.contains{$0.moe != nil})
        try await wait("preparation \(String(describing:host.predictionPreparation?.phase)) host=\(host.statusMessage)"){host.predictionPreparation?.phase == .completed || host.predictionPreparation?.phase == .failed}
        XCTAssertEqual(host.predictionPreparation?.phase,.completed,host.statusMessage)
        XCTAssertNotNil(host.snapshot(for:view),host.statusMessage)
        XCTAssertNotNil(host.snapshot(for:WorkflowExamples.trainAddress)?.ports["routing"],"Captured routing ports must be visible to canvas inspection, independently of prediction rows")
        let result=try XCTUnwrap(host.outcomes.first)
        XCTAssertNil(result.recordError)
        let reference=WorkflowHistoryReference(projectID:host.owner.projectID,directoryID:try XCTUnwrap(UUID(uuidString:result.directory.lastPathComponent)))
        let archived=try await WorkflowHistoryVerificationWork.run{WorkflowHistoryReader.load(reference,root:root)}
        guard case .verified(let verified)=archived else {return XCTFail("Cold MoE History verification failed")}
        XCTAssertEqual(verified.modelContent?.routingTables["routing"]?.rowCount,8)
        XCTAssertTrue(verified.modelContent?.trainingProgress?.points.contains{$0.moe != nil} == true)
        let query=try XCTUnwrap(host.predictionTableViewSession(for:.init(address:WorkflowExamples.trainAddress,port:"routing"),selectionRequired:false))
        XCTAssertTrue(query.load());try await wait{!query.isBusy}
        XCTAssertNotNil(query.page,query.statusMessage)
        query.invalidate()
        let package=try await WorkflowTrainedExportWriter().stage(reference,root:root)
        defer{package.remove()}
        let destination=root.appendingPathComponent("independent-moe")
        try package.publish(to:destination)
        let runtime=try XCTUnwrap(workspace.runtime)
        let finished=expectation(description:"Independent MoE package inference")
        var exitCode:Int32?,lines:[String]=[]
        let worker=try WorkflowExecutionSession.launchWorker(runtime.executableURL,[destination.appendingPathComponent("inference.py").path,"--example"],root,{lines.append($0)},{code,_ in exitCode=code;finished.fulfill()})
        defer{worker.cancel()}
        await fulfillment(of:[finished],timeout:30)
        XCTAssertEqual(exitCode,0,lines.joined(separator:"\n"))
        XCTAssertTrue(lines.last?.contains("logits")==true)
        let routing=outcomeRoutingPath(result.directory)
        try Data("tampered".utf8).write(to:routing)
        let refused=try await WorkflowHistoryVerificationWork.run{WorkflowHistoryReader.load(reference,root:root)}
        if case .verified=refused{XCTFail("Tampered routing must invalidate recorded output")}
    }
    private func outcomeRoutingPath(_ directory:URL)->URL {directory.appendingPathComponent("training/artifacts/routing.json")}
    func testRealOwnedPreflightWithoutOptimizerUpdates()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Needs explicit local runtime")}
        _=NSApplication.shared
        let root=try canonicalBackendTemporaryDirectory(),workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(MixtureExpertsPractice.lessonID,device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession),validator=host.makeMixtureValidationSession(),values=try MixtureExpertsPractice.graph().nodes[1].config
        validator.run(values);try await wait(validator.message){!validator.isBusy}
        XCTAssertEqual(validator.validated,values,validator.message)
        XCTAssertFalse(workspace.isBusy);XCTAssertTrue(host.outcomes.isEmpty)
        try? FileManager.default.removeItem(at:root)
    }
}
