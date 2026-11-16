import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowExplicitConsumerIsolationTests:XCTestCase {
    private func idle(_ host:WorkflowWorkspaceSession)async throws {
        let deadline=Date().addingTimeInterval(60)
        while host.hasActiveOperation && Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
        XCTAssertFalse(host.hasActiveOperation,host.statusMessage)
        try await Task.sleep(for:.milliseconds(150))
    }
    private func output(_ host:WorkflowWorkspaceSession,from:WorkflowEndpoint,root:URL,id:String)throws->URL {
        let address=WorkflowNodeAddress(domain:.data,localID:id),owner=host.owner
        XCTAssertTrue(host.canvas.apply(.insert(.init(address:address,payload:.dataOperator(operatorID:"kwip.data.outputDelimited",config:["filename":.string(id+".csv"),"separator":.string(","),"columns":.array([]),"overwrite":false]))),lease:owner.lease()))
        XCTAssertTrue(host.canvas.apply(.connect(from:from,to:.init(address:address,port:"table")),lease:owner.lease()))
        host.canvas.selectNode(try XCTUnwrap(host.canvas.projection.displayIDByAddress[address]),extending:false)
        host.outputDestinations.pickDirectory={root}
        XCTAssertTrue(host.outputDestinations.chooseDestination(for:address,lease:owner.lease()))
        return root.appendingPathComponent(id+".csv")
    }
    func testDirectEvaluationDoesNotSweepCurrentTrainingOutputs()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else{throw XCTSkip("Needs actual runtime")}
        let root=try canonicalBackendTemporaryDirectory(),workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadPreparedWorkflowExample(device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession),owner=host.owner
        defer{host.cancel();if !workspace.isBusy{try? FileManager.default.removeItem(at:root)}}
        let view=WorkflowNodeAddress(domain:.data,localID:"required-view")
        XCTAssertTrue(host.canvas.apply(.insert(.init(address:view,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))),lease:owner.lease()))
        XCTAssertTrue(host.canvas.apply(.connect(from:.init(address:WorkflowExamples.trainAddress,port:"predictions"),to:.init(address:view,port:"table")),lease:owner.lease()))
        XCTAssertTrue(host.run(train:WorkflowExamples.trainAddress),host.statusMessage);try await idle(host)
        let producer=try XCTUnwrap(host.currentExecution?.runID),priorOperation=host.operationIdentity
        let evaluate=WorkflowNodeAddress(domain:.control,localID:"selected-evaluate")
        XCTAssertTrue(host.canvas.apply(.insert(.init(address:evaluate,payload:.evaluateTest(config:["settingsVersion":2,"partitionRole":.string("validation"),"device":.string("cpu")]))),lease:owner.lease()))
        XCTAssertTrue(host.canvas.apply(.connect(from:.init(address:WorkflowExamples.trainAddress,port:"model"),to:.init(address:evaluate,port:"model")),lease:owner.lease()))
        XCTAssertTrue(host.canvas.apply(.connect(from:.init(address:.init(domain:.data,localID:"fit"),port:"validation"),to:.init(address:evaluate,port:"test")),lease:owner.lease()))
        let destination=try output(host,from:.init(address:WorkflowExamples.trainAddress,port:"predictions"),root:root,id:"unrequested-output")
        XCTAssertTrue(host.evaluate(address:evaluate),host.statusMessage);try await idle(host)
        XCTAssertNotEqual(host.operationIdentity,priorOperation)
        XCTAssertEqual(host.evaluationOutcomes.first?.outcome.status,.completed,host.statusMessage)
        XCTAssertEqual(host.currentExecution?.runID,producer,"Explicit Evaluate does not retrain a current model")
        XCTAssertFalse(FileManager.default.fileExists(atPath:destination.path),"Direct Evaluate must not publish Train's sibling table output")
    }
    func testDirectCanvasGenerationDoesNotPublishConnectedOutput()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else{throw XCTSkip("Needs actual runtime")}
        let root=try canonicalBackendTemporaryDirectory(),workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(TextGenerationPracticeExample.exampleID,device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        defer{host.cancel();if !workspace.isBusy{try? FileManager.default.removeItem(at:root)}}
        XCTAssertTrue(host.run(train:WorkflowExamples.trainAddress),host.statusMessage);try await idle(host)
        let producer=try XCTUnwrap(host.currentExecution?.runID),old=host.operationIdentity
        let destination=try output(host,from:.init(address:TextGenerationPracticeExample.address,port:"table"),root:root,id:"unrequested-generation-output")
        XCTAssertTrue(host.generateText(address:TextGenerationPracticeExample.address),host.statusMessage);try await idle(host)
        XCTAssertNotEqual(host.operationIdentity,old)
        XCTAssertEqual(host.generation?.record?.status,"completed",host.statusMessage)
        XCTAssertEqual(host.currentExecution?.runID,producer)
        XCTAssertFalse(FileManager.default.fileExists(atPath:destination.path),"Direct Generate must not execute a downstream Table Output")
    }
    func testDirectImportedGenerationAndSaveOwnIdentityWithoutPublishingTables()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"],let fixture=ProcessInfo.processInfo.environment["KWIPML_TEST_MODEL_FOLDER"] else{throw XCTSkip("Needs actual runtime and pinned tiny causal model")}
        let root=try canonicalBackendTemporaryDirectory(),workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(FineTuningPracticeExample.exampleID,device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession),owner=host.owner
        defer{host.cancel();if !workspace.isBusy{try? FileManager.default.removeItem(at:root)}}
        for consumer in [FineTuningPracticeExample.generate,FineTuningPracticeExample.save] {
            let edge=try XCTUnwrap(owner.document.edges.first{$0.to.address==consumer && $0.to.port=="model"})
            XCTAssertTrue(host.canvas.apply(.reconnect(edgeID:edge.id,from:.init(address:FineTuningPracticeExample.model,port:"model"),to:.init(address:consumer,port:"model")),lease:owner.lease()))
        }
        try await host.modelLoader.importFolder(URL(fileURLWithPath:fixture),address:FineTuningPracticeExample.model,runtime:try XCTUnwrap(workspace.runtime),policy:workspace.executionPolicy(for:owner.projectID))
        let destination=try output(host,from:.init(address:FineTuningPracticeExample.generate,port:"table"),root:root,id:"unrequested-imported-output")
        let old=host.operationIdentity
        XCTAssertTrue(host.generateText(address:FineTuningPracticeExample.generate),host.statusMessage);try await idle(host)
        XCTAssertNotEqual(host.operationIdentity,old,"Imported generation must own a fresh operation too")
        XCTAssertNotNil(host.modelLoader.outputArtifact(.init(address:FineTuningPracticeExample.generate,port:"table")),host.statusMessage)
        XCTAssertFalse(FileManager.default.fileExists(atPath:destination.path))
        let generated=host.operationIdentity,modelDestination=root.appendingPathComponent("explicit-imported-save")
        XCTAssertTrue(host.modelDestinations.set(destination:modelDestination,for:FineTuningPracticeExample.save,lease:owner.lease()))
        XCTAssertTrue(host.saveModel(address:FineTuningPracticeExample.save),host.statusMessage);try await idle(host)
        XCTAssertNotEqual(host.operationIdentity,generated)
        XCTAssertEqual(host.externalModelSaveReceipts[FineTuningPracticeExample.save]?.path,modelDestination.path)
        XCTAssertTrue(FileManager.default.fileExists(atPath:modelDestination.appendingPathComponent("config.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath:destination.path))
        XCTAssertTrue(host.modelLoader.trainingRecords.isEmpty,"Direct loaded-model consumers must not train unrelated Train")
    }
}


extension MCPWorkflowGatewayTests {
    func testExplicitEvaluateRefusesMissingModelBeforeNativeTrainingFallback()async throws {
        var document=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        document=try WorkflowTrainedModelSource.promoteAuthoring(document)
        let address=WorkflowNodeAddress(domain:.control,localID:"explicit-evaluate")
        document.nodes.append(.init(address:address,payload:.evaluateTest(config:["settingsVersion":2,"partitionRole":.string("validation"),"device":.string("cpu")])))
        document.edges.append(.init(from:.init(address:WorkflowExamples.trainAddress,port:"model"),to:.init(address:address,port:"model"),kind:.frozenModel))
        let f=try fixture(document:document)
        let host=try XCTUnwrap(f.windows.openControllers.first?.session.workspace.workflowSession)
        do {_=try await call(f,.runStart,["terminal":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(address))]);XCTFail("Implicit training accepted")}
        catch let refusal as MCPSettings.Refusal {XCTAssertEqual(refusal.code,"completed_model_required")}
        XCTAssertFalse(host.hasActiveOperation)
        XCTAssertNil(host.currentExecution)
    }
}
