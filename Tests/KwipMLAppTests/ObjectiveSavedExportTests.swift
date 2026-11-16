import XCTest
import Combine
import Darwin
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ObjectiveSavedExportTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func temporaryRoot()throws->URL {
        guard let canonical=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        defer {free(canonical)}
        return URL(fileURLWithPath:String(cString:canonical)).appendingPathComponent(UUID().uuidString)
    }
    private struct Saved {
        let task:LossLearningTask
        let reference:WorkflowHistoryReference
        let directory:URL
        let recipe:LossRecipe
        let outputShape:[Int]
        let registryHash:String
        let workerRunID:String
    }
    private func capture(_ tasks:[LossLearningTask],root:URL,device:String="cpu")async throws->[Saved] {
        let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"].map {URL(fileURLWithPath:$0)} ?? repository
        let workspace=WorkspaceStore(rootOverride:development,storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertEqual(workspace.lossRuntimeCapabilities.componentIDs,LossConfiguration.componentIDs)
        var saved:[Saved]=[]
        for task in tasks {
            let document=try WorkflowExamples.preparedLossObjective(task:task,device:device)
            _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
            XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
            let host=try XCTUnwrap(workspace.workflowSession)
            let resource=try XCTUnwrap(BundledLossComponents.entries[task.componentID]?.learning?.sourceResource)
            let source=root.appendingPathComponent(task.rawValue+device+".csv")
            try FileManager.default.copyItem(at:repository.appendingPathComponent("Sources/KwipMLApp/Resources/"+resource+".csv"),to:source)
            try workspace.sourceAccess.remember(source)
            XCTAssertTrue(host.bindings.setPath(source.path,for:WorkflowExamples.sourceAddress,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
            let finished=expectation(description:"Actual captured objective "+task.rawValue)
            let subscription=host.$outcomes.dropFirst().filter {!$0.isEmpty}.prefix(1).sink {_ in finished.fulfill()}
            XCTAssertTrue(host.run(train:WorkflowExamples.trainAddress),host.statusMessage)
            await fulfillment(of:[finished],timeout:45)
            subscription.cancel()
            if workspace.isBusy {host.cancel()}
            let outcome=try XCTUnwrap(host.outcomes.first)
            XCTAssertEqual(outcome.status,.completed,outcome.error ?? host.statusMessage);XCTAssertNil(outcome.recordError)
            let directoryID=try XCTUnwrap(UUID(uuidString:outcome.directory.lastPathComponent))
            let recipe=try XCTUnwrap(outcome.request.plan.lossRecipe)
            let shape=try WorkflowMappingRequirements.outputContract(outcome.request.plan.model.graph,registry:try XCTUnwrap(outcome.request.capturedModelRegistry)).layout.sample.dimensions
            let recorded=try WorkflowArchiveCodec.decodeOutcome(Data(contentsOf:outcome.directory.appendingPathComponent("outcome.json")))
            saved.append(.init(task:task,reference:.init(projectID:document.projectID,directoryID:directoryID),directory:outcome.directory,recipe:recipe,outputShape:shape,registryHash:outcome.request.modelRegistrySHA256,workerRunID:try XCTUnwrap(recorded.workerRunID)))
            host.retire();workspace.retireWorkflowSession()
        }
        return saved
    }
    func testAllEightActualObjectivesColdReadCPUHistoryAndExportCPUAndMPSPortableWeights()async throws {
        let root=try temporaryRoot()
        defer {try? FileManager.default.removeItem(at:root)}
        for device in ["cpu","mps"] {
            let saved=try await capture(Array(LossLearningTask.allCases),root:root,device:device)
            XCTAssertEqual(saved.count,8,device)
            var replayPaths:[String]=[]
            for item in saved {
                if device == "cpu" {
                    // All original cold History assertions run after capture retired its live host.
                    try SavedObjectiveHistoryAssertions.verify(task:item.task,reference:item.reference,directory:item.directory,
                        recipe:item.recipe,outputShape:item.outputShape,registryHash:item.registryHash,workerRunID:item.workerRunID,root:root)
                }
                let original=try Data(contentsOf:item.directory.appendingPathComponent("training/artifacts/checkpoint.pt"))
                let package=try WorkflowHistoryReader.stageTrainedExport(item.reference,root:root)
                defer {package.remove()}
                let path=package.directory
                let contract=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:path.appendingPathComponent("input-contract.json")))
                XCTAssertEqual(contract["schemaVersion"],.integer(2))
                guard case .object(let output)=contract["output"],case .object(let objective)=contract["objective"] else {return XCTFail("Generic objective export must retain output and training context")}
                XCTAssertEqual(output["kind"],.string("modelOutput"));XCTAssertEqual(output["dtype"],.string("float32"))
                XCTAssertEqual(output["shape"],.array([.string("N")]+item.outputShape.map { .integer(Int64($0)) }))
                XCTAssertNil(output["classIndices"])
                XCTAssertEqual(try LossConfiguration.decodeRecipe(try XCTUnwrap(objective["recipe"])),item.recipe.configuration)
                XCTAssertEqual(objective["scalarization"],.string("mean"))
                let source=try JSONDecoder().decode(WorkflowNodeAddress.self,from:WorkflowArchiveCodec.encode(try XCTUnwrap(objective["source"])))
                let mapping=try JSONDecoder().decode(LossTargetMapping.self,from:WorkflowArchiveCodec.encode(try XCTUnwrap(objective["targetMapping"])))
                XCTAssertEqual(source,item.recipe.source);XCTAssertEqual(mapping,item.recipe.targetMapping)
                XCTAssertEqual(package.inputContract.columns,item.recipe.targetMapping.inputColumns)
                XCTAssertEqual(try Data(contentsOf:path.appendingPathComponent("checkpoint.pt")),original)
                XCTAssertEqual(try Data(contentsOf:path.appendingPathComponent("objective_recipe.py")),try Data(contentsOf:item.directory.appendingPathComponent("training/artifacts/objective_recipe.py")))
                XCTAssertEqual(Set(package.files.map(\.path)),["model.py","checkpoint.pt","objective_recipe.py","input-contract.json","inference.py","README.md","requirements.txt"])
                for name in package.files.map(\.path).filter({$0 != "checkpoint.pt"})+["export-manifest.json"] {
                    let text=try String(contentsOf:path.appendingPathComponent(name),encoding:.utf8)
                    XCTAssertFalse(text.contains(root.path));XCTAssertFalse(text.contains(repository.path))
                    for privateKey in ["trainingRowIDs","validationRowIDs","sourceBindings","observationIDs"] {XCTAssertFalse(text.contains(privateKey))}
                }
                let readme=try String(contentsOf:path.appendingPathComponent("README.md"),encoding:.utf8)
                XCTAssertTrue(readme.contains("objective_recipe.py"));XCTAssertTrue(readme.contains("modelOutput"))
                XCTAssertFalse(readme.contains("Results are two logits"))
                let destination=root.appendingPathComponent("published-"+item.task.rawValue+device)
                try package.publish(to:destination)
                replayPaths.append(destination.path)
                XCTAssertThrowsError(try package.publish(to:destination))
                XCTAssertEqual(try Data(contentsOf:destination.appendingPathComponent("checkpoint.pt")),original)
            }
            // Every objective keeps its independent model/checkpoint/shape/privacy assertions.
            // One interpreter per captured device avoids repeating Torch import sixteen times.
            let replayManifest=root.appendingPathComponent("published-objective-replays-"+device+".json")
            try JSONEncoder().encode(replayPaths).write(to:replayManifest,options:.atomic)
            let probe="""
            import sys,json,torch,importlib.util
            from pathlib import Path
            def verify(root,index):
                spec=importlib.util.spec_from_file_location('portable_'+str(index),root/'inference.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
                assert m.ROOT==root.resolve(), 'Each helper must retain its own artifact root'
                contract=json.loads((root/'input-contract.json').read_text());shape=contract['input']['shape'][1:]
                x=torch.linspace(-.75,.8,5*__import__('math').prod(shape),dtype=torch.float32).reshape(5,*shape)
                ns={'__name__':'independent_model_'+str(index)};exec(compile((root/'model.py').read_bytes(),'model.py','exec'),ns)
                model=ns['Model']().to(dtype=torch.float32);checkpoint=torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['modelState']);model.eval()
                with torch.inference_mode():expected=model(x)
                predictor=m.load_model()
                for batch in (1,5):torch.testing.assert_close(predictor.predict(x,batch_size=batch),expected,rtol=1e-6,atol=1e-7)
                assert list(expected.shape)==[5,*contract['output']['shape'][1:]]
                forbidden={'trainingRowIDs','validationRowIDs','sourcePath','sourceBindings','datasetCSV'}
                def inspect(value):
                    if isinstance(value,dict):
                        assert not(set(value)&forbidden)
                        for k,v in value.items():inspect(k);inspect(v)
                    elif isinstance(value,(list,tuple)):
                        for v in value:inspect(v)
                    elif isinstance(value,str):assert sys.argv[2] not in value
                inspect(checkpoint)
                print('verified generic objective model output')
            paths=json.loads(Path(sys.argv[1]).read_text())
            assert len(paths)==8 and len(set(paths))==8
            for index,path in enumerate(paths):
                verify(Path(path),index)
            print('verified all eight published objective exports')
            """
            let done=expectation(description:"Saved portable objective batch "+device)
            var code:Int32?
            var lines:[String]=[]
            let process=try WorkflowExecutionSession.launchWorker(repository.appendingPathComponent(".venv/bin/python"),["-I","-c",probe,replayManifest.path,root.path],root,{lines.append($0)},{status,_ in code=status;done.fulfill()})
            defer {process.cancel()}
            await fulfillment(of:[done],timeout:30)
            XCTAssertEqual(code,0,device+" "+lines.joined(separator:"\n"))
            XCTAssertEqual(lines.filter {$0 == "verified generic objective model output"}.count,LossLearningTask.allCases.count,device)
            XCTAssertTrue(lines.contains("verified all eight published objective exports"),device)
        }
    }
    func testActualObjectiveExportRetainsAtomicStorageAndTamperGuards()async throws {
        let root=try temporaryRoot()
        defer {try? FileManager.default.removeItem(at:root)}
        let captured=try await capture([.mse],root:root)
        let item=try XCTUnwrap(captured.first)
        XCTAssertThrowsError(try WorkflowHistoryReader.stageTrainedExport(item.reference,root:root,storageLimitBytes:1))
        let package=try WorkflowHistoryReader.stageTrainedExport(item.reference,root:root)
        defer {package.remove()}
        try Data("changed helper".utf8).write(to:package.directory.appendingPathComponent("objective_recipe.py"))
        let destination=root.appendingPathComponent("must-not-exist")
        XCTAssertThrowsError(try package.publish(to:destination));XCTAssertFalse(FileManager.default.fileExists(atPath:destination.path))
        let original=item.directory.appendingPathComponent("training/artifacts/objective_recipe.py")
        try Data("changed captured helper".utf8).write(to:original)
        XCTAssertThrowsError(try WorkflowHistoryReader.stageTrainedExport(item.reference,root:root))
    }
}
