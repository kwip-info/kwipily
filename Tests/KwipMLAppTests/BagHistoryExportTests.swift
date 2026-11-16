import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class BagHistoryExportTests:XCTestCase {
    func testPacketContractPreservesActualRolesMappingAndVocabularyWithoutDenseAxes()throws {
        let fixture=BagWorkflowArchiveTests(),registry=try fixture.registry()
        for packed in [false,true] {for weighted in [false,true] {for objective in [false,true] {
            let plan=try fixture.compile(fixture.fixture(packed:packed,weighted:weighted,objective:objective))
            let mapping=try XCTUnwrap(plan.bagInput)
            let input=try TrainedModelInputContract(graph:plan.model.graph,registry:registry,bagMapping:mapping,targetColumn:plan.targetColumn,recipe:plan.lossRecipe)
            XCTAssertEqual(input.input["kind"],.string("bagPacket"))
            XCTAssertEqual(input.input["schemaVersion"],1)
            XCTAssertEqual(input.input["sourceID"],.string("bags"))
            XCTAssertEqual(input.input["mapping"],mapping.value)
            XCTAssertEqual(input.input["numEmbeddings"],11)
            XCTAssertEqual(input.input["preprocessing"],.string("alreadyPrepared"))
            XCTAssertNil(input.input["shape"]);XCTAssertNil(input.input["dtype"]);XCTAssertNil(input.input["featureColumns"])
            guard case .object(let ports)=input.input["ports"] else {return XCTFail("Actual packet roles required")}
            XCTAssertEqual(Set(ports.keys),Set(["indices"]+(packed ? ["offsets"]:[])+(weighted ? ["per_sample_weights"]:[])))
            XCTAssertEqual(ports["indices"],.object(["role":.string("indices"),"dtype":.string("int64")]))
            if weighted {XCTAssertEqual(ports["per_sample_weights"],.object(["role":.string("per_sample_weights"),"dtype":.string("float32")]))}
            XCTAssertEqual(input.columns,["i0","i1"])
            XCTAssertTrue(input.reviewSummary.contains("group"))
            XCTAssertTrue(input.valueGuidance.contains(packed ? "empty":"exactly"))
            if objective {XCTAssertTrue(input.outputReviewSummary.contains("numeric"))}
        }}}
    }
    func testPacketContractRejectsMappingControlsAndTargetOverlap()throws {
        let fixture=BagWorkflowArchiveTests(),registry=try fixture.registry()
        let fixed=try fixture.compile(fixture.fixture()),packed=try fixture.compile(fixture.fixture(packed:true,weighted:true))
        XCTAssertThrowsError(try TrainedModelInputContract(graph:fixed.model.graph,registry:registry,bagMapping:XCTUnwrap(packed.bagInput),targetColumn:"label",recipe:nil))
        XCTAssertThrowsError(try TrainedModelInputContract(graph:fixed.model.graph,registry:registry,bagMapping:XCTUnwrap(fixed.bagInput),targetColumn:"i0",recipe:nil))
    }
    func testActualBinaryTrainedExportStagesPinnedPacketContract()async throws {try await exerciseExport(objective:false)}
    func testActualSelectedObjectiveTrainedExportStagesHelperAndPacketContract()async throws {try await exerciseExport(objective:true)}
    private func exerciseExport(objective:Bool)async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Enable the actual development runtime")}
        let repository=URL(fileURLWithPath:development)
        guard let physical=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        defer {free(physical)}
        let root=URL(fileURLWithPath:String(cString:physical)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let runtime=try XCTUnwrap(workspace.runtime)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.embeddingBagRuntimeCapabilities.isComplete)
        let proof=try XCTUnwrap(workspace.moduleCompositionDiscoveryEvidence)
        let registry=try ComponentRegistry.builtIn()
        XCTAssertEqual(try JSONDecoder().decode(ComponentRegistry.self,from:Data(contentsOf:runtime.registryURL)),registry)
        var document=try PreparedWorkflowExample.component("embedding_bag").document(device:"cpu")
        if objective {
            let mapping=LossTargetMapping(componentID:"loss_cross_entropy",inputColumns:["id_a","id_b","id_c"],roles:[.init(role:"target",columns:["label"],sampleShape:[],dtype:.int64)])
            let mappingIndex=try XCTUnwrap(document.nodes.firstIndex {$0.address == .init(domain:.control,localID:"mapping")})
            document.nodes[mappingIndex].payload = .trainingData(config:["bagInput":.object(["schemaVersion":1,"indicesColumns":.array(["id_a","id_b","id_c"].map(JSONValue.string))]),"lossTargetMapping":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(mapping))])
            document=try WorkflowTrainRecipes.migrateAuthoringSchema(document)
            let train=try XCTUnwrap(document.nodes.firstIndex {$0.address == WorkflowExamples.trainAddress})
            guard case .train(var settings)=document.nodes[train].payload else {return XCTFail("Missing Train")}
            settings["objective"]=try LossConfiguration(componentID:"loss_cross_entropy",values:[:]).recipeValue
            document.nodes[train].payload = .train(config:settings)
            XCTAssertFalse(document.edges.contains {$0.kind == .objectiveRecipe})
        }
        XCTAssertEqual(PreparedWorkflowExample.component("embedding_bag").observationsResource,"bag-observations")
        let library=workspace.projectLibrary
        _ = try library.saveSource(.workflow(document),id:document.projectID)
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:registry,dataRegistry:.load(from:repository.appendingPathComponent("runner/dataflow-operators.json")))
        let budget=try ResolvedExecutionBudget.resolve(global:.init(maximumRunSeconds:60),projectMaximumRunSeconds:nil,resources:.init(cpuThreads:2,maximumResidentBytes:1_073_741_824))
        let request=try WorkflowExecutionRequest.capture(from:owner,train:WorkflowExamples.trainAddress,runtime:runtime,executionBudget:budget,sampleCapture:.factory,moduleCompositionEvidence:proof)
        let id=UUID(),directory=root.appendingPathComponent("workflow-runs/\(document.projectID)/\(id.uuidString)")
        let ended=expectation(description:"Published example completes")
        var outcome:WorkflowExecutionOutcome?,acquired=0,released=0
        let reservation=WorkflowExecutionReservation(acquire:{acquired+=1;return UUID()},release:{_ in released+=1})
        let run=WorkflowExecutionSession(request:request,owner:owner,sources:[WorkflowExamples.sourceAddress:try .capture(repository.appendingPathComponent("Sources/KwipMLApp/Resources/bag-observations.csv"))],directory:directory,reservation:reservation,onTerminal:{outcome=$0;ended.fulfill()})
        XCTAssertTrue(run.start())
        await fulfillment(of:[ended],timeout:65)
        defer {if owner.isBusy {run.cancel()}}
        XCTAssertEqual(run.phase,.completed,run.statusMessage+" / "+(outcome?.error ?? ""))
        XCTAssertEqual(acquired,1);XCTAssertEqual(released,1);XCTAssertFalse(owner.isBusy)
        XCTAssertNil(outcome?.error);XCTAssertNil(outcome?.recordError)
        XCTAssertNotNil(outcome?.modelManifestSHA256)
        XCTAssertEqual(run.metrics.map(\.epoch),[1,2,3])
        let preview=try XCTUnwrap(run.bagPredictionPreview)
        XCTAssertEqual(preview.evaluationCount,16);XCTAssertEqual(preview.rows.count,16)
        XCTAssertTrue(preview.rows.allSatisfy {$0.input.indices.count == 3 && $0.input.perSampleWeights == nil})
        XCTAssertNil(run.predictionPreview);XCTAssertNil(run.objectivePredictionPreview)
        let captures=try XCTUnwrap(run.modelCaptures)
        for origin in request.plan.model.origins.values {XCTAssertNotNil(captures.ports(instancePath:origin.instancePath,localNodeID:origin.localNodeID))}
        let cold=try XCTUnwrap(WorkflowHistoryReader.loadModelContent(.init(projectID:document.projectID,directoryID:id),root:root))
        XCTAssertEqual(cold.bagPredictionPreview,preview)
        XCTAssertEqual(cold.metrics.map(\.trainLoss),run.metrics.map(\.trainLoss))
        XCTAssertNotEqual(cold.outcome.runID,cold.outcome.workerRunID,"Workflow and worker identities are distinct; packet context belongs to the worker.")
        let reference=WorkflowHistoryReference(projectID:document.projectID,directoryID:id)
        let package=try WorkflowHistoryReader.stageTrainedExport(reference,root:root)
        defer {package.remove()}
        let contract=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:package.directory.appendingPathComponent("input-contract.json")))
        XCTAssertEqual(contract["schemaVersion"],3)
        XCTAssertEqual(Set(contract.keys),Set(objective ? ["schemaVersion","input","output","objective"]:["schemaVersion","input","output"]))
        XCTAssertEqual(package.inputContract.input["kind"],.string("bagPacket"))
        XCTAssertEqual(package.files.contains {$0.path == "objective_recipe.py"},objective)
        XCTAssertFalse(package.files.contains {$0.path.contains("predictions") || $0.path.contains("prepared") || $0.path.hasSuffix("csv") || $0.path.contains("derivative-results")})
        let readme=try String(contentsOf:package.directory.appendingPathComponent("README.md"),encoding:.utf8)
        XCTAssertTrue(readme.contains("whole-group"));XCTAssertFalse(readme.contains("pass a CPU int64 tensor"))
        let portable=root.appendingPathComponent("portable-model")
        try package.publish(to:portable)
        let expected:JSONValue = .array(preview.rows.map {row in
            var record:[String:JSONValue]=["input":.object(["indices":.array(row.input.indices.map(JSONValue.integer))])]
            if let predicted=row.predicted {record["predicted"] = .integer(Int64(predicted))}
            if let output=row.output {record["output"] = .array(output.values.map(JSONValue.number))}
            return .object(record)
        })
        try WorkflowArchiveCodec.encode(expected).write(to:root.appendingPathComponent("expected.json"))
        try Data("corrupt".utf8).write(to:directory.appendingPathComponent("training/artifacts/checkpoint.pt"))
        XCTAssertThrowsError(try WorkflowHistoryReader.stageTrainedExport(reference,root:root))
        try FileManager.default.removeItem(at:root.appendingPathComponent("workflow-runs"))
        package.remove()
        let script=root.appendingPathComponent("verify-export.py")
        try #"""
        import json,sys
        from pathlib import Path
        import torch
        root=Path(__file__).parent
        assert not (root/'workflow-runs').exists()
        assert all(not (Path(p)/'runner'/'dataflow.py').exists() for p in sys.path if p)
        expected=json.loads((root/'expected.json').read_text())
        sys.path.insert(0,str(root/'portable-model'))
        from inference import load_model
        predictor=load_model()
        records=[row['input'] for row in expected]
        first=predictor.predict(records,batch_size=1)
        second=predictor.predict(records,batch_size=7)
        torch.testing.assert_close(first,second,rtol=1e-5,atol=1e-6)
        if 'output' in expected[0]:
            torch.testing.assert_close(first,torch.tensor([row['output'] for row in expected],dtype=torch.float32),rtol=1e-5,atol=1e-6)
        else:
            assert first.argmax(1).tolist()==[row['predicted'] for row in expected]
        print(json.dumps({'verified':True,'rows':len(records),'batchSizes':[1,7]}))
        """#.write(to:script,atomically:true,encoding:.utf8)
        let verified=expectation(description:"Isolated relocated actual trained export")
        var exitCode:Int32?,lines:[String]=[]
        let process=try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:"/usr/bin/env"),["-u","PYTHONPATH","-u","PYTHONHOME","PYTHONNOUSERSITE=1",runtime.executableURL.path,script.path],root,{lines.append($0)},{code,error in exitCode=code;if !error.isEmpty {lines.append(error)};verified.fulfill()})
        defer {process.cancel()}
        await fulfillment(of:[verified],timeout:45)
        XCTAssertEqual(exitCode,0,lines.joined(separator:"\n"))
        XCTAssertTrue(lines.contains {$0.contains("\"verified\": true")},lines.joined(separator:"\n"))

    }
}
