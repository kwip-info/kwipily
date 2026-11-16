import XCTest
import Darwin
import KwipMLCore
@testable import KwipMLApp

/// Real universal workflow, cold History, portable helper retention and CPU epoch-boundary continuation.
@MainActor final class WorkflowDataLoadingIntegrationTests:XCTestCase {
    func testSelectedReaderAndBatchingSurviveHistoryExportAndExactResume()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Enable the actual development runtime")}
        guard let pointer=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        let root=URL(fileURLWithPath:String(cString:pointer)).appendingPathComponent(UUID().uuidString);free(pointer)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let repo=URL(fileURLWithPath:development),workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.dataLoadingRuntimeCapabilities.contains("dataset_source"));XCTAssertTrue(workspace.dataLoadingRuntimeCapabilities.contains("data_loader"))
        let runtime=try XCTUnwrap(workspace.runtime),data=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        var base=try WorkflowExamples.preparedParallelClassifier(device:"cpu");base.schemaVersion=13
        let readerValues:[String:JSONValue]=["kind":.string("subset"),"indices":.array([4,1,0,7,3]),"max_rows":100000]
        var loaderValues=DataLoaderConfiguration.defaults;loaderValues["batch_size"]=2;loaderValues["drop_last"]=true
        let trainIndex=try XCTUnwrap(base.nodes.firstIndex {$0.address == WorkflowExamples.trainAddress})
        guard case .train(var training)=base.nodes[trainIndex].payload else {return XCTFail("Missing Train")}
        training["datasetAccess"] = .object(readerValues);training["batching"] = .object(loaderValues)
        base.nodes[trainIndex].payload = .train(config:training)
        func document(_ epochs:Int,_ mode:WorkflowTrainingMode)->WorkflowDocument {
            var copy=base
            for index in copy.nodes.indices {
                if case .runSetup(var fields)=copy.nodes[index].payload {fields["epochs"] = .integer(Int64(epochs));copy.nodes[index].payload = .runSetup(config:fields)}
                if case .train(var fields) = copy.nodes[index].payload {fields["trainingMode"] = .string(mode.rawValue);copy.nodes[index].payload = .train(config:fields)}
            }
            return copy
        }
        let source=try WorkflowCapturedSource.capture(repo.appendingPathComponent("Sources/KwipMLApp/Resources/workflow-observations.csv"))
        let budget=try ResolvedExecutionBudget.resolve(global:.init(maximumRunSeconds:60),projectMaximumRunSeconds:nil,resources:.init(cpuThreads:2,maximumResidentBytes:1_073_741_824))
        func execute(_ value:WorkflowDocument,_ label:String,snapshot:WorkflowCheckpointSnapshot?=nil,parent:WorkflowCheckpointReference?=nil)async throws->WorkflowExecutionOutcome {
            let library=ProjectLibrary(directory:root.appendingPathComponent("projects-"+label));_ = try library.saveSource(.workflow(value),id:value.projectID)
            let owner=try WorkflowAuthoringStore(library:library,projectID:value.projectID,modelRegistry:workspace.registry,dataRegistry:data)
            let request=try WorkflowExecutionRequest.capture(from:owner,train:WorkflowExamples.trainAddress,runtime:runtime,checkpointParent:parent,executionBudget:budget,sampleCapture:.factory,moduleCompositionEvidence:workspace.moduleCompositionDiscoveryEvidence)
            XCTAssertEqual(request.plan.datasetRecipe?.configuration.indices,[4,1,0,7,3]);XCTAssertEqual(request.plan.loaderRecipe?.configuration.batchSize,2)
            let ended=expectation(description:label);var outcome:WorkflowExecutionOutcome?
            let directory=root.appendingPathComponent("workflow-runs/\(value.projectID)/\(UUID().uuidString)")
            let run=WorkflowExecutionSession(request:request,owner:owner,sources:[WorkflowExamples.sourceAddress:source],directory:directory,reservation:.init(acquire:{UUID()},release:{_ in}),checkpointSnapshot:snapshot,onTerminal:{outcome=$0;ended.fulfill()})
            guard run.start() else {throw WorkflowArchiveError(run.statusMessage)}
            defer {if owner.isBusy {run.cancel()}}
            await fulfillment(of:[ended],timeout:65)
            let result=try XCTUnwrap(outcome)
            XCTAssertEqual(result.status,.completed,result.error ?? run.statusMessage);XCTAssertNil(result.recordError)
            guard result.status == .completed else {throw WorkflowArchiveError(result.error ?? run.statusMessage)}
            return result
        }
        let parent=try await execute(document(2,.fresh),"parent")
        let reference=WorkflowHistoryReference(projectID:base.projectID,directoryID:try XCTUnwrap(UUID(uuidString:parent.directory.lastPathComponent)))
        guard case .verified(let verified)=WorkflowHistoryReader.load(reference,root:root) else {return XCTFail("Selected reader parent must verify cold")}
        XCTAssertEqual(verified.plan.datasetRecipe?.source,WorkflowExamples.trainAddress);XCTAssertEqual(verified.plan.loaderRecipe?.source,WorkflowExamples.trainAddress)
        let parentFolder=parent.directory.appendingPathComponent("training/artifacts")
        var parentBytes:[String:Data]=[:]
        for name in ["manifest.json","checkpoint.pt","metrics.json","data_loading_recipe.py"] {parentBytes[name]=try Data(contentsOf:parentFolder.appendingPathComponent(name))}
        let manifest=try JSONDecoder().decode([String:JSONValue].self,from:try XCTUnwrap(parentBytes["manifest.json"]))
        guard case .object(let settings)=manifest["settings"] else {return XCTFail("Missing actual training settings")}
        XCTAssertEqual(settings["batchesPerEpoch"],2,"Five selected rows with final short group dropped yield exactly two batches")
        XCTAssertNotNil(verified.modelContent?.trainingProgress)
        let package=try WorkflowHistoryReader.stageTrainedExport(reference,root:root)
        defer {package.remove()}
        XCTAssertEqual(package.files.first {$0.path == "data_loading_recipe.py"}?.sha256,WorkflowArchiveCodec.hash(try XCTUnwrap(parentBytes["data_loading_recipe.py"])))
        XCTAssertEqual(try Data(contentsOf:package.directory.appendingPathComponent("data_loading_recipe.py")),parentBytes["data_loading_recipe.py"])
        let portable=root.appendingPathComponent("portable");try package.publish(to:portable)
        let selected=WorkflowCheckpointReference(projectID:base.projectID,directoryID:reference.directoryID.uuidString,runID:parent.runID.uuidString,requestIdentity:parent.request.requestIdentity,modelManifestSHA256:try XCTUnwrap(parent.modelManifestSHA256))
        let snapshot=try WorkflowHistoryReader.stageCheckpoint(selected,root:root);defer {snapshot.remove()}
        XCTAssertNotNil(snapshot.dataLoadingIdentity)
        _ = try snapshot.parentRecord()
        let resumed=try await execute(document(3,.exactResume),"resumed",snapshot:snapshot,parent:selected)
        let uninterrupted=try await execute(document(3,.fresh),"uninterrupted")
        for (name,bytes) in parentBytes {XCTAssertEqual(try Data(contentsOf:parentFolder.appendingPathComponent(name)),bytes,"Continuation changed its immutable parent: "+name)}
        let resumedReference=WorkflowHistoryReference(projectID:base.projectID,directoryID:try XCTUnwrap(UUID(uuidString:resumed.directory.lastPathComponent)))
        guard case .verified(let child)=WorkflowHistoryReader.load(resumedReference,root:root) else {return XCTFail("CPU exact-resume child must verify cold")}
        XCTAssertEqual(child.plan.trainingMode,.exactResume);XCTAssertNotNil(child.plan.datasetRecipe);XCTAssertNotNil(child.plan.loaderRecipe)
        try Data(contentsOf:resumed.directory.appendingPathComponent("training/artifacts/checkpoint.pt")).write(to:root.appendingPathComponent("resumed.pt"))
        try Data(contentsOf:uninterrupted.directory.appendingPathComponent("training/artifacts/checkpoint.pt")).write(to:root.appendingPathComponent("full.pt"))
        let script=root.appendingPathComponent("verify-loading.py")
        try #"""
        import importlib.util,json
        from pathlib import Path
        import torch
        root=Path(__file__).parent
        a=torch.load(root/'resumed.pt',map_location='cpu',weights_only=True)
        b=torch.load(root/'full.pt',map_location='cpu',weights_only=True)
        assert a['epoch']==b['epoch']==3
        def equal(x,y):
            if isinstance(x,torch.Tensor):assert torch.equal(x,y)
            elif isinstance(x,dict):
                assert set(x)==set(y)
                for key in x:equal(x[key],y[key])
            elif isinstance(x,(tuple,list)):
                assert len(x)==len(y)
                for left,right in zip(x,y):equal(left,right)
            else:assert x==y
        equal(a['modelState'],b['modelState']);equal(a['optimizerState'],b['optimizerState'])
        spec=importlib.util.spec_from_file_location('portable_inference',root/'portable'/'inference.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        result=helper.load_model().predict(torch.tensor([[.25,-.5],[1.,2.]],dtype=torch.float32))
        assert list(result.shape)==[2,2] and torch.isfinite(result).all()
        print('selected-reader-exact-resume-and-export-ok')
        """#.write(to:script,atomically:true,encoding:.utf8)
        let checked=expectation(description:"Actual learned CPU continuation and selected-reader portable export")
        var exitCode:Int32?,lines:[String]=[]
        let process=try WorkflowExecutionSession.launchWorker(runtime.executableURL,["-I",script.path],root,{lines.append($0)},{code,message in exitCode=code;if !message.isEmpty {lines.append(message)};checked.fulfill()})
        defer {process.cancel()}
        await fulfillment(of:[checked],timeout:45)
        XCTAssertEqual(exitCode,0,lines.joined(separator:"\n"))
    }
}
