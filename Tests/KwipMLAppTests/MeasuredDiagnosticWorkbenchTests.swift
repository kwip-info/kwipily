import XCTest
import KwipMLCore
@testable import KwipMLApp

final class MeasuredDiagnosticWorkbenchTests:XCTestCase {
    func testNamedCalibrationCorpusCoversEveryRuleWithHealthyControls()throws {
        let fixtures=MeasuredDiagnosticCalibration.fixtures
        XCTAssertGreaterThanOrEqual(fixtures.count,27)
        XCTAssertEqual(Set(fixtures.map(\.id)).count,fixtures.count)
        let results=try MeasuredDiagnosticCalibration.evaluate()
        XCTAssertEqual(Set(results.map(\.ruleID)),Set(MeasuredDiagnostics.rules.map(\.id)))
        for result in results {
            XCTAssertGreaterThanOrEqual(result.positives,1,result.ruleID)
            XCTAssertGreaterThanOrEqual(result.negatives,2,result.ruleID)
            XCTAssertEqual(result.falsePositives,0,result.ruleID)
            XCTAssertEqual(result.falseNegatives,0,result.ruleID)
            XCTAssertEqual(result.withheldControls,0,result.ruleID)
        }
    }
    func testSuppressionRequiresReasonScopeExpiryAndReactivatesForChangedEvidence()throws {
        let now=Date(timeIntervalSince1970:1_800_000_000)
        let report=try MeasuredDiagnostics.evaluate(positive())
        let finding=try XCTUnwrap(report.findings.first)
        let suppression=try MeasuredDiagnosticSuppression(report:report,finding:finding,reason:"Known stationary control",now:now,expiresAt:now.addingTimeInterval(3600))
        XCTAssertTrue(suppression.applies(to:report,finding:finding,at:now))
        XCTAssertFalse(suppression.applies(to:report,finding:finding,at:now.addingTimeInterval(3601)))
        var input=positive();input.rawError="Changed captured evidence"
        let changed=try MeasuredDiagnostics.evaluate(input)
        XCTAssertFalse(suppression.applies(to:changed,finding:try XCTUnwrap(changed.findings.first),at:now))
        XCTAssertThrowsError(try MeasuredDiagnosticSuppression(report:report,finding:finding,reason:" ",now:now,expiresAt:now.addingTimeInterval(3600)))
        XCTAssertThrowsError(try MeasuredDiagnosticSuppression(report:report,finding:finding,reason:"Review later",now:now,expiresAt:now.addingTimeInterval(90*86400)))
    }
    func testRedactedExportOmitsRawErrorsPrivateNamesAndScalarSamples()throws {
        var input=positive();input.rawError="MPS out of memory reading /Users/alice/private.csv"
        input.activations=[.init(nodeID:"patient-alice-private",component:"tanh",values:Array(repeating:0.9999999,count:8))]
        let report=try MeasuredDiagnostics.evaluate(input)
        let data=try MeasuredDiagnosticExport.redacted(report)
        let text=try XCTUnwrap(String(data:data,encoding:.utf8))
        XCTAssertFalse(text.contains("alice"));XCTAssertFalse(text.contains("/Users/"));XCTAssertFalse(text.contains("0.9999999"))
        XCTAssertTrue(text.contains("runtime.memory.v1"));XCTAssertTrue(text.contains("redacted"))
    }
    @MainActor func testLearningRateRepairIsExplicitUndoableAndRejectsStaleEvidence()throws {
        let root=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:root)}
        let (owner,source)=try makeOwner(root)
        var input=positive();input.workflowSHA256=WorkflowArchiveCodec.hash(try WorkflowCodec.encode(source))
        let report=try MeasuredDiagnostics.evaluate(input),finding=try XCTUnwrap(report.findings.first)
        let repair=try MeasuredDiagnosticRepair.propose(input:input,findingID:finding.id,train:WorkflowExamples.trainAddress,owner:owner,journalRoot:root)
        XCTAssertEqual(owner.document,source,"Proposing must not change the model")
        XCTAssertEqual(repair.proposedLearningRate,repair.originalLearningRate/2)
        try repair.apply(to:owner)
        XCTAssertNotEqual(owner.document,source);XCTAssertTrue(owner.canUndo)
        XCTAssertThrowsError(try repair.apply(to:owner),"An accepted proposal cannot apply twice")
        try repair.revert(in:owner)
        XCTAssertEqual(owner.document.nodes,source.nodes)
        XCTAssertEqual(owner.document.edges,source.edges)
        let stale=try makeOwner(root.appendingPathComponent("stale"))
        var staleInput=positive();staleInput.workflowSHA256=WorkflowArchiveCodec.hash(try WorkflowCodec.encode(stale.1))
        XCTAssertTrue(stale.0.rename("Changed after capture",lease:stale.0.lease()))
        XCTAssertThrowsError(try MeasuredDiagnosticRepair.propose(input:staleInput,findingID:finding.id,train:WorkflowExamples.trainAddress,owner:stale.0,journalRoot:root))
        let records=try FileManager.default.contentsOfDirectory(at:root.appendingPathComponent("diagnostic-repairs/\(owner.projectID)"),includingPropertiesForKeys:nil)
        XCTAssertGreaterThanOrEqual(records.count,3,"Intent, accepted change and revert must be retained")
    }
    @MainActor func testRepairDoesNotOverwriteEditsMadeAfterProposal()throws {
        let root=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:root)}
        let (owner,source)=try makeOwner(root)
        var input=positive();input.workflowSHA256=WorkflowArchiveCodec.hash(try WorkflowCodec.encode(source))
        let finding=try XCTUnwrap(MeasuredDiagnostics.evaluate(input).findings.first)
        let repair=try MeasuredDiagnosticRepair.propose(input:input,findingID:finding.id,train:WorkflowExamples.trainAddress,owner:owner,journalRoot:root)
        XCTAssertTrue(owner.rename("Later user edit",lease:owner.lease()))
        let after=owner.document
        XCTAssertThrowsError(try repair.apply(to:owner));XCTAssertEqual(owner.document,after)
    }
    @MainActor func testRepairRefusesAnotherTrainEvenWhenWholeWorkflowMatches()throws {
        let root=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:root)}
        let (owner,source)=try makeOwner(root,secondTrain:true)
        var input=positive();input.workflowSHA256=WorkflowArchiveCodec.hash(try WorkflowCodec.encode(source))
        let finding=try XCTUnwrap(MeasuredDiagnostics.evaluate(input).findings.first)
        let other=WorkflowNodeAddress(domain:.control,localID:"second-train")
        XCTAssertThrowsError(try MeasuredDiagnosticRepair.propose(input:input,findingID:finding.id,train:other,owner:owner,journalRoot:root))
        XCTAssertEqual(owner.document,source)
    }
    @MainActor func testInlineTrainRepairPreservesOtherSettingsAndJournalsFullUndoableChange()throws {
        let root=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:root)}
        let(owner,source)=try makeOwner(root,currentRunSettings:true)
        XCTAssertEqual(source.schemaVersion,16)
        XCTAssertFalse(source.nodes.contains {if case .runSetup=$0.payload {return true};return false})
        let train=WorkflowExamples.trainAddress
        guard case .train(let before)=try XCTUnwrap(source.nodes.first {$0.address==train}).payload else {return XCTFail("Train fixture")}
        let run=try XCTUnwrap(WorkflowTrainRunSetup.configuration(before,schemaVersion:16))
        XCTAssertEqual(run["learningRate"]?.doubleValue,0.03)
        var input=positive();input.workflowSHA256=WorkflowArchiveCodec.hash(try WorkflowCodec.encode(source))
        let finding=try XCTUnwrap(MeasuredDiagnostics.evaluate(input).findings.first)
        let repair=try MeasuredDiagnosticRepair.propose(input:input,findingID:finding.id,train:train,owner:owner,journalRoot:root)
        XCTAssertEqual(repair.target,train);XCTAssertEqual(owner.document,source)
        XCTAssertEqual(repair.proposedLearningRate,0.015)
        try repair.apply(to:owner)
        guard case .train(let after)=try XCTUnwrap(owner.document.nodes.first {$0.address==train}).payload else {return XCTFail("Train fixture")}
        var expectedRun=run;expectedRun["learningRate"] = .number(0.015)
        var expected=before;expected["runSetup"] = .object(expectedRun)
        XCTAssertEqual(after,expected)
        XCTAssertEqual(after["trainingMode"],.string("fineTune"))
        XCTAssertEqual(after["batching"],before["batching"])
        XCTAssertEqual(owner.document.edges,source.edges)
        XCTAssertEqual(owner.document.nodes.filter {$0.address != train},source.nodes.filter {$0.address != train})
        let journal=root.appendingPathComponent("diagnostic-repairs/\(owner.projectID)")
        let files=try FileManager.default.contentsOfDirectory(at:journal,includingPropertiesForKeys:nil)
        let receipts=try files.map {try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:$0))}
        let applied=try XCTUnwrap(receipts.first {$0["action"] == .string("applied")})
        XCTAssertEqual(try WorkflowArchiveCodec.encode(XCTUnwrap(applied["before"])),try WorkflowArchiveCodec.encode(JSONValue.object(before)))
        XCTAssertEqual(try WorkflowArchiveCodec.encode(XCTUnwrap(applied["proposed"])),try WorkflowArchiveCodec.encode(JSONValue.object(expected)))
        XCTAssertEqual(applied["target"],.object(["domain":.string("control"),"localID":.string("train")]))
        try repair.revert(in:owner)
        XCTAssertEqual(owner.document.nodes,source.nodes);XCTAssertEqual(owner.document.edges,source.edges)
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath:journal.path).contains {$0.hasSuffix("-reverted.json")})
    }
    @MainActor func testInlineAndConnectedOptimizersRefuseInactiveFallbackRepairWithoutMutationOrAudit()throws {
        let root=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:root)}
        for authority in ["inline","connected"] {
            let directory=root.appendingPathComponent(authority)
            let(owner,source)=try makeOwner(directory,currentRunSettings:authority == "inline",optimizerAuthority:authority)
            var input=positive();input.workflowSHA256=WorkflowArchiveCodec.hash(try WorkflowCodec.encode(source))
            let finding=try XCTUnwrap(MeasuredDiagnostics.evaluate(input).findings.first)
            XCTAssertThrowsError(try MeasuredDiagnosticRepair.propose(input:input,findingID:finding.id,train:WorkflowExamples.trainAddress,owner:owner,journalRoot:directory)) {error in
                XCTAssertTrue(error.localizedDescription.contains("optimizer owns the active learning rate"),error.localizedDescription)
            }
            XCTAssertEqual(owner.document,source);XCTAssertFalse(owner.canUndo)
            XCTAssertFalse(FileManager.default.fileExists(atPath:directory.appendingPathComponent("diagnostic-repairs").path))
        }
    }
    @MainActor func testIndependentSuppressionReadersPreserveEachOthersWrites()throws {
        let root=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let project=UUID().uuidString,now=Date()
        let first=MeasuredDiagnosticSuppressionStore(root:root,projectID:project),second=MeasuredDiagnosticSuppressionStore(root:root,projectID:project)
        let a=try MeasuredDiagnostics.evaluate(positive());var changed=positive();changed.runID=UUID().uuidString
        let b=try MeasuredDiagnostics.evaluate(changed)
        let one=try MeasuredDiagnosticSuppression(report:a,finding:XCTUnwrap(a.findings.first),reason:"First review",now:now,expiresAt:now.addingTimeInterval(3600))
        let two=try MeasuredDiagnosticSuppression(report:b,finding:XCTUnwrap(b.findings.first),reason:"Second review",now:now,expiresAt:now.addingTimeInterval(3600))
        try first.save(one,now:now);try second.save(two,now:now)
        first.reload();XCTAssertEqual(Set(first.entries.map(\.id)),[one.id,two.id])
        try second.remove(two.id);first.reload();XCTAssertEqual(first.entries,[one])
    }
    private func positive()->MeasuredDiagnosticInput {
        var input=MeasuredDiagnosticInput(runID:"D50BE2A0-5F47-4BF6-A86A-2002292F3F10",requestSHA256:String(repeating:"b",count:64),metrics:[])
        input.capturedTrain=WorkflowExamples.trainAddress
        input.gradients=(1...10).map{.init(epoch:$0,batch:0,presentElements:10,nonfiniteElements:0,zeroElements:0,missingParameters:0,l2Norm:$0<=5 ? 0.01:2)}
        return input
    }
    @MainActor private func makeOwner(_ root:URL,secondTrain:Bool=false,currentRunSettings:Bool=false,optimizerAuthority:String?=nil)throws->(WorkflowAuthoringStore,WorkflowDocument) {
        let library=ProjectLibrary(directory:root.appendingPathComponent("projects"))
        var source=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        if currentRunSettings {
            source=try WorkflowTrainRunSetup.migrate(source)
            let index=try XCTUnwrap(source.nodes.firstIndex {$0.address==WorkflowExamples.trainAddress})
            guard case .train(var config)=source.nodes[index].payload else {throw DataFlowContractError("Train fixture")}
            var batches=DataLoaderConfiguration.defaults;batches["batch_size"]=7
            config["batching"] = .object(batches);config["trainingMode"] = .string("fineTune")
            source.nodes[index].payload = .train(config:config)
        }
        if let optimizerAuthority {
            let recipe=try OptimizerConfiguration(componentID:"optimizer_sgd",values:["lr":.number(0.07)])
            if optimizerAuthority == "inline" {
                let index=try XCTUnwrap(source.nodes.firstIndex {$0.address==WorkflowExamples.trainAddress})
                guard case .train(var config)=source.nodes[index].payload else {throw DataFlowContractError("Train fixture")}
                config["optimizer"]=recipe.recipeValue;source.nodes[index].payload = .train(config:config)
            } else {
                let optimizer=WorkflowNodeAddress(domain:.control,localID:"optimizer")
                source.schemaVersion=max(source.schemaVersion,6)
                source.nodes.append(.init(address:optimizer,payload:.optimizerRecipe(type:recipe.componentID,configurationVersion:1,config:recipe.values)))
                source.edges.append(.init(from:.init(address:optimizer,port:"out"),to:.init(address:WorkflowExamples.trainAddress,port:"optimizer"),kind:.optimizerRecipe))
            }
        }
        if secondTrain {
            var train=try XCTUnwrap(source.nodes.first{$0.address==WorkflowExamples.trainAddress})
            train.address = .init(domain:.control,localID:"second-train");source.nodes.append(train)
            source.edges += source.edges.filter{$0.to.address==WorkflowExamples.trainAddress}.map { edge in
                var copy=edge;copy.id=UUID().uuidString;copy.to.address=train.address;return copy
            }
        }
        source=try WorkflowCodec.decode(WorkflowCodec.encode(source))
        _=try library.saveSource(.workflow(source))
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return(try WorkflowAuthoringStore(library:library,projectID:source.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))),source)
    }
}
