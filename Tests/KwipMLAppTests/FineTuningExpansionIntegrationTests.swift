import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class FineTuningExpansionIntegrationTests:XCTestCase {
    private func wait(_ condition:()->Bool)async throws {
        let end=Date().addingTimeInterval(50)
        while Date()<end {if condition(){return};try await Task.sleep(for:.milliseconds(15))}
        throw WorkflowArchiveError("Timed out waiting for actual expansion workflow.")
    }
    func testClosingInspectionDuringPreparationDoesNotStartDeferredModelTask()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"],let fixture=ProcessInfo.processInfo.environment["KWIPML_TEST_MODEL_FOLDER"]else{throw XCTSkip("Needs reviewed local runtime and tiny safe model")}
        let root=try canonicalBackendTemporaryDirectory(),workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace);XCTAssertTrue(workspace.loadComponentWorkflowExample(FineTuningPracticeExample.exampleID,device:"cpu"))
        let host=try XCTUnwrap(workspace.workflowSession),runtime=try XCTUnwrap(workspace.runtime),policy=try workspace.executionPolicy(for:host.owner.projectID)
        defer{host.cancel();if !workspace.isBusy{try? FileManager.default.removeItem(at:root)}}
        try await host.modelLoader.importFolder(URL(fileURLWithPath:fixture),address:FineTuningPracticeExample.model,runtime:runtime,policy:policy)
        XCTAssertTrue(host.modelDestinations.set(destination:root.appendingPathComponent("unused"),for:FineTuningPracticeExample.save,lease:host.owner.lease()))
        XCTAssertTrue(host.owner.select(.init(nodeAddresses:[FineTuningPracticeExample.train]),lease:host.owner.lease()))
        let context=try WorkflowControlNodeContext(owner:host.owner,address:FineTuningPracticeExample.train)
        XCTAssertTrue(host.inspectFineTuning(FineTuningPracticeExample.train,draft:context.configuration),host.statusMessage)
        XCTAssertTrue(host.isInspectingFineTuning(FineTuningPracticeExample.train))
        host.cancelFineTuningInspection(FineTuningPracticeExample.train)
        try await wait{!workspace.isBusy && !host.hasActiveOperation && !host.modelLoader.isBusy}
        XCTAssertFalse(host.isInspectingFineTuning(FineTuningPracticeExample.train));XCTAssertNil(host.modelLoader.fineTuningPreview);XCTAssertTrue(host.modelLoader.trainingRecords.isEmpty)
    }
    func testStopKeepsCommittedCPURecoveryWithoutPublishingInference()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"],let fixture=ProcessInfo.processInfo.environment["KWIPML_TEST_MODEL_FOLDER"]else{throw XCTSkip("Needs explicit reviewed local runtime and tiny safetensors model")}
        let root=try canonicalBackendTemporaryDirectory(),workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(FineTuningPracticeExample.exampleID,device:"cpu"))
        let host=try XCTUnwrap(workspace.workflowSession),runtime=try XCTUnwrap(workspace.runtime),policy=try workspace.executionPolicy(for:host.owner.projectID)
        defer{host.cancel();if !workspace.isBusy{try? FileManager.default.removeItem(at:root)}}
        XCTAssertTrue(host.owner.select(.init(nodeAddresses:[FineTuningPracticeExample.train]),lease:host.owner.lease()))
        let context=try WorkflowControlNodeContext(owner:host.owner,address:FineTuningPracticeExample.train);var config=context.configuration
        if case .object(var fine)=config["fineTuning"]{fine["checkpointCount"]=2;fine["checkpointBytes"]=100_000_000;fine["mode"] = .string("lora");fine["targets"] = .array([.string("attentionInput")]);fine["rank"]=2;config["fineTuning"] = .object(fine)}
        if case .object(var run)=config["runSetup"]{run["epochs"]=500;config["runSetup"] = .object(run)}
        XCTAssertTrue(context.authority.configure(config,expected:context.configuration))
        try await host.modelLoader.importFolder(URL(fileURLWithPath:fixture),address:FineTuningPracticeExample.model,runtime:runtime,policy:policy)
        let destination=root.appendingPathComponent("must-not-exist")
        XCTAssertTrue(host.modelDestinations.set(destination:destination,for:FineTuningPracticeExample.save,lease:host.owner.lease()))
        XCTAssertTrue(host.owner.select(.init(),lease:host.owner.lease()));XCTAssertTrue(host.run(),host.statusMessage)
        try await wait{
            guard let id=host.modelLoader.trainingRunID else{return false}
            let folder=root.appendingPathComponent("workflow-operations/"+host.owner.projectID+"/"+id+"/output/training/checkpoints")
            return ((try? FileManager.default.contentsOfDirectory(atPath:folder.path)) ?? []).contains{!$0.hasPrefix(".") && FileManager.default.fileExists(atPath:folder.appendingPathComponent($0+"/result-manifest.json").path)}
        }
        host.cancel();try await wait{!workspace.isBusy && !host.modelLoader.isBusy && !host.hasActiveOperation}
        XCTAssertTrue(host.modelLoader.trainingRecords.isEmpty);XCTAssertFalse(FileManager.default.fileExists(atPath:destination.path))
        let recovery=try XCTUnwrap(host.modelLoader.fineTuningRecoveries.first,host.modelLoader.status)
        let boundary=try await Task.detached{try recovery.capture(root:root)}.value
        XCTAssertEqual(boundary.report["continuation"],.string("cpuEpochBoundary"));XCTAssertGreaterThan(boundary.report["completedEpoch"]?.integerValue ?? 0,0)
        XCTAssertEqual(host.modelLoader.trainingPhase,"cancelled")
        let restored=ModelFineTuningRecovery.scan(root:root,project:host.owner.projectID)
        XCTAssertEqual(restored.first?.id,recovery.id)
    }
    func testPromptResponsePauseEvaluatePortableResumeAndIndependentReload()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"],let fixture=ProcessInfo.processInfo.environment["KWIPML_TEST_MODEL_FOLDER"]else{throw XCTSkip("Needs explicit reviewed local runtime and tiny safetensors model")}
        let root=try canonicalBackendTemporaryDirectory()
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(FineTuningPracticeExample.exampleID,device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession),runtime=try XCTUnwrap(workspace.runtime),policy=try workspace.executionPolicy(for:host.owner.projectID)
        defer{host.cancel();if !workspace.isBusy{try? FileManager.default.removeItem(at:root)}}
        let csv=root.appendingPathComponent("responses.csv")
        try Data(("prompt,response\n"+String(repeating:"hello,world .\nworld,hello .\n",count:40)).utf8).write(to:csv)
        XCTAssertTrue(host.bindings.setPath(csv.path,for:FineTuningPracticeExample.source,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        XCTAssertTrue(host.owner.select(.init(nodeAddresses:[FineTuningPracticeExample.train]),lease:host.owner.lease()))
        let train=try WorkflowControlNodeContext(owner:host.owner,address:FineTuningPracticeExample.train)
        var config=train.configuration
        config["fineTuning"] = .object(try FineTuningConfiguration(["mode":.string("lora"),"targets":.array([.string("attentionInput")]),"rank":2,"alpha":4,"maximumTokens":8,"stride":7,"exampleFormat":.string("promptResponse"),"promptColumn":.string("prompt"),"responseColumn":.string("response"),"separator":.string(" "),"lossCoverage":.string("response"),"checkpointCount":2,"checkpointBytes":100_000_000]).values)
        if case .object(var run)=config["runSetup"]{run["epochs"]=5;config["runSetup"] = .object(run)}
        XCTAssertTrue(train.authority.configure(config,expected:train.configuration),train.statusMessage)
        try await host.modelLoader.importFolder(URL(fileURLWithPath:fixture),address:FineTuningPracticeExample.model,runtime:runtime,policy:policy)
        XCTAssertTrue(host.modelDestinations.set(destination:root.appendingPathComponent("standalone"),for:FineTuningPracticeExample.save,lease:host.owner.lease()))
        XCTAssertTrue(host.owner.select(.init(),lease:host.owner.lease()))
        XCTAssertTrue(host.run(),host.statusMessage)
        try await wait{host.modelLoader.trainingAddress==FineTuningPracticeExample.train && host.modelLoader.isBusy}
        host.modelLoader.pauseFineTuning()
        try await wait{!host.hasActiveOperation && !workspace.isBusy}
        let first=try XCTUnwrap(host.modelLoader.trainingRecords.first,host.statusMessage+" / "+host.modelLoader.status)
        let report=try await Task.detached{try first.report(root:root)}.value
        XCTAssertEqual(report["paused"],true);XCTAssertLessThan(try XCTUnwrap(report["completedEpoch"]?.integerValue),5)
        XCTAssertNotNil(host.externalModelSaveReceipts[FineTuningPracticeExample.save],host.statusMessage)
        XCTAssertNotNil(host.modelLoader.generationRecord,"Pause still publishes its selected inference model for downstream consumers")
        let evaluate=WorkflowNodeAddress(domain:.control,localID:"causal-score")
        XCTAssertTrue(host.canvas.apply(.insert(.init(address:evaluate,payload:.evaluateTest(config:["device":.string("cpu"),"batchSize":2]))),lease:host.owner.lease()))
        XCTAssertTrue(host.canvas.apply(.connect(from:.init(address:FineTuningPracticeExample.train,port:"model"),to:.init(address:evaluate,port:"model")),lease:host.owner.lease()))
        XCTAssertTrue(host.canvas.apply(.connect(from:.init(address:FineTuningPracticeExample.partition,port:"test"),to:.init(address:evaluate,port:"test")),lease:host.owner.lease()))
        let view=WorkflowNodeAddress(domain:.data,localID:"score-view")
        XCTAssertTrue(host.canvas.apply(.insert(.init(address:view,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))),lease:host.owner.lease()))
        XCTAssertTrue(host.canvas.apply(.connect(from:.init(address:evaluate,port:"predictions"),to:.init(address:view,port:"table")),lease:host.owner.lease()))
        let previousPreparation=host.predictionPreparation
        let previousTrainingIDs=host.modelLoader.trainingRecords.map(\.id)
        XCTAssertTrue(host.evaluate(address:evaluate),host.statusMessage)
        try await wait{!workspace.isBusy && !host.modelLoader.isBusy && !host.hasActiveOperation}
        XCTAssertTrue(host.predictionPreparation === previousPreparation,"Explicit Evaluate must not admit a downstream data sweep")
        XCTAssertNil(host.snapshot(for:view),"A newly connected View requires its own explicit admission")
        let scored=try XCTUnwrap(host.modelLoader.currentEvaluation,host.statusMessage+" / "+host.modelLoader.status)
        XCTAssertGreaterThan(host.modelLoader.evaluationSummary?["scoredTokens"]?.integerValue ?? 0,0)
        XCTAssertNotNil(host.modelLoader.predictionArtifact(evaluate));XCTAssertNotNil(host.snapshot(for:evaluate))
        XCTAssertTrue(host.prepareData(at:view),host.statusMessage)
        try await wait{!workspace.isBusy && !host.modelLoader.isBusy && !host.hasActiveOperation}
        XCTAssertEqual(host.scopedPreparation?.phase,.completed,host.statusMessage)
        XCTAssertNotNil(host.snapshot(for:view),"Explicit View must consume the captured causal predictions")
        XCTAssertEqual(host.modelLoader.currentEvaluation?.id,scored.id,"Viewing captured predictions must not reevaluate")
        XCTAssertEqual(host.modelLoader.trainingRecords.map(\.id),previousTrainingIDs,"Viewing captured predictions must not retrain")
        let exposureRoot=workspace.applicationDirectory
        let independent=try await Task.detached{try scored.disclose(root:root,exposureRoot:exposureRoot)}.value
        XCTAssertEqual(independent["loss"],host.modelLoader.evaluationSummary?["loss"])
        XCTAssertTrue(HistoryTableCatalog.rows(causalEvaluations:[scored]).contains{$0.target == .causalEvaluation(scored.id)})
        XCTAssertTrue(host.owner.select(.init(nodeAddresses:[FineTuningPracticeExample.save]),lease:host.owner.lease()))
        let save=try WorkflowControlNodeContext(owner:host.owner,address:FineTuningPracticeExample.save)
        var saveConfig=save.configuration;saveConfig["content"] = .string("continuation")
        XCTAssertTrue(save.authority.configure(saveConfig,expected:save.configuration))
        let package=root.appendingPathComponent("cpu-resume")
        XCTAssertTrue(host.modelDestinations.set(destination:package,for:FineTuningPracticeExample.save,lease:host.owner.lease()))
        XCTAssertTrue(host.saveModel(address:FineTuningPracticeExample.save),host.statusMessage)
        try await wait{!host.modelLoader.isBusy && !host.hasActiveOperation}
        XCTAssertTrue(FileManager.default.fileExists(atPath:package.appendingPathComponent("latest/kwip-continuation.json").path),host.modelLoader.status)
        XCTAssertTrue(FileManager.default.fileExists(atPath:package.appendingPathComponent("base/model.safetensors").path))
        let base=try XCTUnwrap(host.modelLoader.reference(FineTuningPracticeExample.model))
        let imported=try await Task.detached{try ModelImportedResume.importPackage(package,base:base,root:root,accountingRoot:root,limit:nil)}.value
        try FileManager.default.removeItem(at:package)
        _=try await Task.detached{try imported.capture(root:root,base:base)}.value
        host.modelLoader.reloadTraining();try await wait{host.modelLoader.importedResumes.contains{$0.id==imported.id}}
        XCTAssertTrue(host.owner.select(.init(nodeAddresses:[FineTuningPracticeExample.train]),lease:host.owner.lease()))
        let trainAgain=try WorkflowControlNodeContext(owner:host.owner,address:FineTuningPracticeExample.train)
        var resumeConfig=trainAgain.configuration
        if case .object(var fine)=resumeConfig["fineTuning"]{fine["startMode"] = .string("resume");fine["resumeRecord"] = .string(imported.id);fine["resumeEpoch"]=0;resumeConfig["fineTuning"] = .object(fine)}
        XCTAssertTrue(trainAgain.authority.configure(resumeConfig,expected:trainAgain.configuration))
        XCTAssertTrue(host.owner.select(.init(nodeAddresses:[FineTuningPracticeExample.save]),lease:host.owner.lease()))
        let saveAgain=try WorkflowControlNodeContext(owner:host.owner,address:FineTuningPracticeExample.save)
        saveConfig=saveAgain.configuration;saveConfig["content"] = .string("full");XCTAssertTrue(saveAgain.authority.configure(saveConfig,expected:saveAgain.configuration),saveAgain.statusMessage)
        XCTAssertTrue(host.modelDestinations.set(destination:root.appendingPathComponent("resumed-standalone"),for:FineTuningPracticeExample.save,lease:host.owner.lease()))
        XCTAssertTrue(host.owner.select(.init(),lease:host.owner.lease()))
        XCTAssertTrue(host.run(),host.statusMessage)
        try await wait{!host.hasActiveOperation && !workspace.isBusy}
        let resumed=try XCTUnwrap(host.modelLoader.trainingRecords.first,host.statusMessage)
        XCTAssertNotEqual(resumed.id,first.id,host.statusMessage+" / "+host.modelLoader.status)
        let final=try await Task.detached{try resumed.report(root:root)}.value
        XCTAssertEqual(final["completedEpoch"],5);XCTAssertEqual(final["initialEpoch"],report["completedEpoch"])
        XCTAssertNotNil(host.externalModelSaveReceipts[FineTuningPracticeExample.save],host.statusMessage)
        _=try await Task.detached{try first.verify(root:root)}.value
    }
}
