import XCTest
import Darwin
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowInlineTrainingRecipesTests:XCTestCase {
    private var verifiedWorkspace:WorkspaceStore?
    private let roles:Set<String>=["optimizer","scheduler","objective","initialization","gradientPolicy","trainability"]
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func configured(_ epochs:Int=2,_ mode:WorkflowTrainingMode = .fresh)throws->WorkflowDocument {
        var document=try WorkflowExamples.preparedLossObjective(task:.mse,device:"cpu")
        let data=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        let baseline=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:data,contractVersion:12)
        let owner=try XCTUnwrap(baseline.model.graph.nodes.first {$0.type == "linear"})
        let optimizer=try OptimizerConfiguration(componentID:"optimizer_sgd",values:["lr":0.02,"momentum":0.3])
        let scheduler=try SchedulerConfiguration(componentID:"scheduler_step_lr",values:["step_size":1,"gamma":0.9])
        let objective=try XCTUnwrap(baseline.lossRecipe).configuration
        let initialization=try InitializationConfiguration(values:["seed":13,"rules":.array([.object(["owner":.object(["node":.string(owner.id),"parameter":.string("weight")]),"algorithm":.string("normal"),"settings":.object(["mean":0.1,"std":0.2])])])])
        let gradient=try GradientPolicyConfiguration(values:["accumulation_steps":2,"gradient_scale":.string("sum"),"requests":.array([])])
        let trainability=try TrainabilityConfiguration(values:["owners":.array([.object(["node":.string(owner.id),"parameter":.string("bias"),"mode":.string("frozen")])])])
        document.nodes.removeAll {if case .lossRecipe = $0.payload {return true};return false}
        document.edges.removeAll {$0.kind == .objectiveRecipe}
        document=try WorkflowTrainRecipes.migrateAuthoringSchema(document)
        for index in document.nodes.indices {
            if case .runSetup(var config)=document.nodes[index].payload {config["epochs"] = .integer(Int64(epochs));document.nodes[index].payload = .runSetup(config:config)}
            if case .train = document.nodes[index].payload {
                document.nodes[index].payload = .train(config:["trainingMode":.string(mode.rawValue),"optimizer":optimizer.recipeValue,"scheduler":scheduler.recipeValue,"objective":objective.recipeValue,"initialization":initialization.recipeValue,"gradientPolicy":gradient.recipeValue,"trainability":trainability.recipeValue])
            }
        }
        return document
    }
    private func makeRequest(_ document:WorkflowDocument,root:URL,parent:WorkflowCheckpointReference?=nil,upgradeRetired:Bool=false)async throws->(WorkflowAuthoringStore,WorkflowExecutionRequest) {
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let workspace:WorkspaceStore
        if let existing=verifiedWorkspace {workspace=existing}
        else {
            workspace=WorkspaceStore(rootOverride:repository,storageOverride:root.appendingPathComponent("runtime-verification"),detectCapabilities:false)
            await verifyRuntimeForIntegration(workspace)
            verifiedWorkspace=workspace
        }
        let data=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        let library=ProjectLibrary(directory:root.appendingPathComponent("projects"))
        _ = try library.saveSource(.workflow(document),id:document.projectID)
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:workspace.registry,dataRegistry:data)
        if upgradeRetired {
            let record=try owner.currentProjectRecord(),originalURL=URL(fileURLWithPath:record.path)
            let originalBytes=try Data(contentsOf:originalURL),revision=owner.document.revision
            XCTAssertTrue(try owner.migrateRetiredToolsForAuthoring())
            let backup=originalURL.appendingPathExtension("before-retired-tools-r\(revision)")
            XCTAssertEqual(try Data(contentsOf:backup),originalBytes,"Exact migration must retain recoverable original bytes")
            XCTAssertEqual(owner.document.revision,revision+1)
            XCTAssertFalse(RetiredTrainingRecipeMigration.requiresUpgrade(owner.document))
        }
        let runtime=try XCTUnwrap(workspace.runtime)
        XCTAssertNotNil(workspace.moduleCompositionDiscoveryEvidence,"Use actual discovered native composition evidence; never synthesize its descriptor.")
        let request=try WorkflowExecutionRequest.capture(from:owner,train:WorkflowExamples.trainAddress,runtime:runtime,checkpointParent:parent,executionBudget:ExecutionPreferences.factory.resolve().budget,derivativeCapture:.factory,moduleCompositionEvidence:workspace.moduleCompositionDiscoveryEvidence)
        return (owner,request)
    }
    private func temporary()throws->URL {
        guard let canonical=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        defer {free(canonical)}
        let root=URL(fileURLWithPath:String(cString:canonical)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        return root
    }
    func testAllSixInlineRecipesCaptureTheirExactTrainSourceAndCanonicalEnvelopes()async throws {
        let root=try temporary();defer {try? FileManager.default.removeItem(at:root)}
        let document=try configured(),(owner,captured)=try await makeRequest(document,root:root)
        defer {owner.invalidate()}
        XCTAssertEqual(captured.plan.inlineTrainingRecipeRoles,roles)
        XCTAssertEqual(document.modelSourceVersions?["model"],2,"Moving controls into Train must preserve the existing model's source semantics.")
        let plan=captured.plan
        XCTAssertEqual([plan.optimizerRecipe?.source,plan.schedulerRecipe?.source,plan.lossRecipe?.source,plan.initializationRecipe?.source,plan.gradientPolicyRecipe?.source,plan.trainabilityRecipe?.source].compactMap {$0},Array(repeating:WorkflowExamples.trainAddress,count:6))
        let directory=root.appendingPathComponent(UUID().uuidString);try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        let source=try WorkflowCapturedSource.capture(repository.appendingPathComponent("Sources/KwipMLApp/Resources/loss-mse-observations.csv"))
        _ = try WorkflowArchiveCaptureWriter.write(request:captured,runID:UUID(),sources:[WorkflowExamples.sourceAddress:source],directory:directory,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        let bytes=try Data(contentsOf:directory.appendingPathComponent("capture.json"))
        let restored=try WorkflowArchiveCodec.decodeCapture(bytes)
        XCTAssertTrue(try restored.plan.matches(plan,compilerContractVersion:12))
        let object=try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [String:Any])
        let archivePlan=try XCTUnwrap(object["plan"] as? [String:Any])
        XCTAssertEqual(archivePlan["inlineTrainingRecipeRoles"] as? [String],roles.sorted())
        XCTAssertEqual(try WorkflowArchiveCodec.encodeCapture(restored),bytes)
        XCTAssertEqual(restored.plan.gradientPolicyRecipe?.configuration,plan.gradientPolicyRecipe?.configuration.recipeValue)
        XCTAssertEqual(restored.plan.trainabilityRecipe?.configuration,plan.trainabilityRecipe?.configuration.recipeValue)
    }
    func testCapturedTrainSourceRequiresExactNonSpuriousInlineMarkers()async throws {
        let root=try temporary();defer {try? FileManager.default.removeItem(at:root)}
        let document=try configured(),(owner,request)=try await makeRequest(document,root:root);defer {owner.invalidate()}
        let directory=root.appendingPathComponent(UUID().uuidString);try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        let source=try WorkflowCapturedSource.capture(repository.appendingPathComponent("Sources/KwipMLApp/Resources/loss-mse-observations.csv"))
        _ = try WorkflowArchiveCaptureWriter.write(request:request,runID:UUID(),sources:[WorkflowExamples.sourceAddress:source],directory:directory,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        let original=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:directory.appendingPathComponent("capture.json"))) as? [String:Any])
        for mutation in ["missing","unmarkedOptimizer","unknown","duplicate","missingRecipe","foreignTrain"] {
            var altered=original,plan=try XCTUnwrap(original["plan"] as? [String:Any])
            switch mutation {
            case "missing":plan.removeValue(forKey:"inlineTrainingRecipeRoles")
            case "unmarkedOptimizer":plan["inlineTrainingRecipeRoles"]=roles.subtracting(["optimizer"]).sorted()
            case "unknown":plan["inlineTrainingRecipeRoles"]=roles.union(["loader"]).sorted()
            case "duplicate":plan["inlineTrainingRecipeRoles"]=roles.sorted()+["optimizer"]
            case "missingRecipe":plan.removeValue(forKey:"optimizerRecipe")
            default:
                var optimizer=try XCTUnwrap(plan["optimizerRecipe"] as? [String:Any])
                optimizer["source"]=["domain":"control","localID":"another-train"]
                plan["optimizerRecipe"]=optimizer
            }
            altered["plan"]=plan
            XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(JSONSerialization.data(withJSONObject:altered)),mutation)
        }
    }
    func testExternalOptimizerUpgradesExactlyWithoutChangingFiveInlineRolesOrOriginalBackup()async throws {
        let root=try temporary();defer {try? FileManager.default.removeItem(at:root)}
        var document=try configured()
        let train=try XCTUnwrap(document.nodes.firstIndex {$0.address == WorkflowExamples.trainAddress})
        guard case .train(var config)=document.nodes[train].payload else {return XCTFail("Missing Train")}
        let optimizer=try OptimizerConfiguration.decodeRecipe(try XCTUnwrap(config.removeValue(forKey:"optimizer")))
        document.nodes[train].payload = .train(config:config)
        let external=WorkflowNodeAddress(domain:.control,localID:"external-optimizer")
        document.nodes.append(.init(address:external,payload:.optimizerRecipe(type:optimizer.componentID,configurationVersion:1,config:optimizer.values)))
        document.edges.append(.init(from:.init(address:external,port:"out"),to:.init(address:WorkflowExamples.trainAddress,port:"optimizer"),kind:.optimizerRecipe))
        let (owner,captured)=try await makeRequest(document,root:root,upgradeRetired:true);defer {owner.invalidate()}
        XCTAssertEqual(captured.plan.inlineTrainingRecipeRoles,roles)
        XCTAssertEqual(captured.plan.optimizerRecipe?.source,WorkflowExamples.trainAddress)
        XCTAssertEqual(captured.plan.optimizerRecipe?.configuration,optimizer)
        let migratedTrain=try XCTUnwrap(owner.document.nodes.first {$0.address==WorkflowExamples.trainAddress})
        guard case .train(let migrated)=migratedTrain.payload else {return XCTFail("Migrated Train")}
        XCTAssertEqual(migrated.filter {$0.key != "optimizer" && $0.key != "runSetup" && $0.key != "dataMapping"},config,"The five existing inline authorities must retain their exact settings")
        XCTAssertEqual(migrated["runSetup"],.object(try WorkflowTrainRunSetup.resolve(in:document,train:WorkflowExamples.trainAddress).configuration))
        XCTAssertFalse(owner.document.nodes.contains {$0.address==external})
        XCTAssertFalse(owner.document.edges.contains {$0.kind == .optimizerRecipe})
        let archive=try WorkflowArchivePlan(captured.plan,compilerContractVersion:12)
        let restored=try JSONDecoder().decode(WorkflowArchivePlan.self,from:WorkflowArchiveCodec.encode(archive))
        XCTAssertTrue(try restored.matches(captured.plan,compilerContractVersion:12))
        // The removed external address cannot impersonate the captured inline Train origin.
        let directory=root.appendingPathComponent(UUID().uuidString);try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        let source=try WorkflowCapturedSource.capture(repository.appendingPathComponent("Sources/KwipMLApp/Resources/loss-mse-observations.csv"))
        _ = try WorkflowArchiveCaptureWriter.write(request:captured,runID:UUID(),sources:[WorkflowExamples.sourceAddress:source],directory:directory,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        var raw=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:directory.appendingPathComponent("capture.json"))) as? [String:Any])
        var alteredPlan=try XCTUnwrap(raw["plan"] as? [String:Any])
        var forged=try XCTUnwrap(alteredPlan["optimizerRecipe"] as? [String:Any])
        forged["source"]=["domain":"control","localID":external.localID]
        alteredPlan["optimizerRecipe"]=forged;raw["plan"]=alteredPlan
        XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(JSONSerialization.data(withJSONObject:raw)))
    }
    func testActualCPUAllSixColdHistoryExportAndExactResumeMatchUninterruptedRun()async throws {
        try await verifyActualCPULifecycle(layerOwned:false)
    }
    func testActualCPULayerSettingsColdHistoryExportAndExactResumeMatchUninterruptedRun()async throws {
        try await verifyActualCPULifecycle(layerOwned:true)
    }
    private func verifyActualCPULifecycle(layerOwned:Bool)async throws {
        guard ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] != nil else {throw XCTSkip("Enable the actual development worker for inline recipe lifecycle verification.")}
        let root=try temporary();defer {try? FileManager.default.removeItem(at:root)}
        var baseline=try configured()
        if layerOwned {
            let train=try XCTUnwrap(baseline.nodes.firstIndex {$0.address==WorkflowExamples.trainAddress})
            guard case .train(var config)=baseline.nodes[train].payload else {return XCTFail("Missing Train")}
            let start=try InitializationConfiguration.decodeRecipe(try XCTUnwrap(config.removeValue(forKey:"initialization")))
            let flags=try TrainabilityConfiguration.decodeRecipe(try XCTUnwrap(config.removeValue(forKey:"trainability")))
            baseline.nodes[train].payload = .train(config:config)
            for rule in start.rules {
                let index=try XCTUnwrap(baseline.nodes.firstIndex {$0.address.localID==rule.owner.node && $0.address.domain == .model})
                var settings=baseline.nodes[index].parameterSettings ?? [:]
                settings[rule.owner.parameter] = .object(["initialization":.object(["algorithm":.string(rule.algorithm),"settings":.object(rule.settings)])])
                baseline.nodes[index].parameterSettings=settings
            }
            for case .object(let flag) in flags.owners {
                let index=try XCTUnwrap(baseline.nodes.firstIndex {$0.address.localID==flag["node"]?.stringValue && $0.address.domain == .model})
                var settings=baseline.nodes[index].parameterSettings ?? [:]
                settings[flag["parameter"]!.stringValue!] = .object(["trainability":flag["mode"]!]);baseline.nodes[index].parameterSettings=settings
            }
            baseline.schemaVersion=15
        }
        let source=try WorkflowCapturedSource.capture(repository.appendingPathComponent("Sources/KwipMLApp/Resources/loss-mse-observations.csv"))
        func document(_ epochs:Int,_ mode:WorkflowTrainingMode)throws->WorkflowDocument {
            var value=baseline
            for index in value.nodes.indices {
                if case .runSetup(var config)=value.nodes[index].payload {config["epochs"] = .integer(Int64(epochs));value.nodes[index].payload = .runSetup(config:config)}
                if case .train(var config)=value.nodes[index].payload {config["trainingMode"] = .string(mode.rawValue);value.nodes[index].payload = .train(config:config)}
            }
            return value
        }
        func execute(_ value:WorkflowDocument,_ label:String,snapshot:WorkflowCheckpointSnapshot?=nil,parent:WorkflowCheckpointReference?=nil)async throws->(WorkflowExecutionOutcome,WorkflowExecutionSession) {
            let (owner,request)=try await makeRequest(value,root:root.appendingPathComponent(label),parent:parent)
            XCTAssertEqual(request.plan.inlineTrainingRecipeRoles,roles)
            let ended=expectation(description:label);var outcome:WorkflowExecutionOutcome?
            let directory=root.appendingPathComponent("workflow-runs/\(value.projectID)/\(UUID().uuidString)")
            let session=WorkflowExecutionSession(request:request,owner:owner,sources:[WorkflowExamples.sourceAddress:source],directory:directory,reservation:.init(acquire:{UUID()},release:{_ in}),checkpointSnapshot:snapshot,onTerminal:{outcome=$0;ended.fulfill()})
            defer {if owner.isBusy {session.cancel()};owner.invalidate()}
            guard session.start() else {throw WorkflowArchiveError(session.statusMessage)}
            await fulfillment(of:[ended],timeout:65)
            let result=try XCTUnwrap(outcome)
            XCTAssertEqual(result.status,.completed,result.error ?? session.statusMessage)
            XCTAssertNil(result.recordError)
            guard result.status == .completed else {throw WorkflowArchiveError(result.error ?? session.statusMessage)}
            return (result,session)
        }
        let (parent,_)=try await execute(document(2,.fresh),"parent")
        let reference=WorkflowHistoryReference(projectID:baseline.projectID,directoryID:try XCTUnwrap(UUID(uuidString:parent.directory.lastPathComponent)))
        guard case .verified(let archived)=WorkflowHistoryReader.load(reference,root:root) else {return XCTFail("All-six inline parent must cold-reopen from captured source.")}
        XCTAssertEqual(archived.plan.inlineTrainingRecipeRoles,roles)
        XCTAssertEqual(archived.plan.optimizerRecipe?.source,WorkflowExamples.trainAddress)
        XCTAssertEqual(archived.plan.initializationRecipe?.source,WorkflowExamples.trainAddress)
        let package=try WorkflowHistoryReader.stageTrainedExport(reference,root:root);defer {package.remove()}
        let portable=root.appendingPathComponent("portable");try package.publish(to:portable)
        let manifest=try JSONDecoder().decode([String:JSONValue].self,from:package.manifestBytes)
        for key in ["schedulerRecipe","initializationRecipe","gradientPolicy","trainability"] {
            XCTAssertNotNil(manifest[key],"Portable provenance must retain selected inline recipe: "+key)
        }
        for helper in ["training_recipe.py","objective_recipe.py","initialization_recipe.py","gradient_policy.py","trainability_recipe.py"] {
            XCTAssertTrue(package.files.contains {$0.path == helper},"Selected inline control helper is absent: "+helper)
        }
        let selected=WorkflowCheckpointReference(projectID:baseline.projectID,directoryID:reference.directoryID.uuidString,runID:parent.runID.uuidString,requestIdentity:parent.request.requestIdentity,modelManifestSHA256:try XCTUnwrap(parent.modelManifestSHA256))
        let snapshot=try WorkflowHistoryReader.stageCheckpoint(selected,root:root);defer {snapshot.remove()}
        let parentCheckpoint=try Data(contentsOf:parent.directory.appendingPathComponent("training/artifacts/checkpoint.pt"))
        let (resumed,resumedSession)=try await execute(document(4,.exactResume),"resumed",snapshot:snapshot,parent:selected)
        let (full,fullSession)=try await execute(document(4,.fresh),"full")
        XCTAssertEqual(resumedSession.metrics.map(\.trainLoss),fullSession.metrics.map(\.trainLoss))
        XCTAssertEqual(resumedSession.metrics.map(\.validationLoss),fullSession.metrics.map(\.validationLoss))
        XCTAssertEqual(try Data(contentsOf:parent.directory.appendingPathComponent("training/artifacts/checkpoint.pt")),parentCheckpoint,"Continuation must not modify its parent.")
        let childReference=WorkflowHistoryReference(projectID:baseline.projectID,directoryID:try XCTUnwrap(UUID(uuidString:resumed.directory.lastPathComponent)))
        guard case .verified(let child)=WorkflowHistoryReader.load(childReference,root:root) else {return XCTFail("Inline exact-resume child must cold-verify.")}
        XCTAssertEqual(child.plan.inlineTrainingRecipeRoles,roles)
        XCTAssertEqual(child.plan.trainingMode,.exactResume)
        let script=root.appendingPathComponent("verify-inline.py")
        try #"""
        import importlib.util,json,sys
        from pathlib import Path
        import torch
        root=Path(__file__).parent
        a,b,parent=map(Path,sys.argv[1:])
        left=torch.load(a/'checkpoint.pt',weights_only=True,map_location='cpu')
        right=torch.load(b/'checkpoint.pt',weights_only=True,map_location='cpu')
        previous=torch.load(parent/'checkpoint.pt',weights_only=True,map_location='cpu')
        def equal(x,y):
            if isinstance(x,torch.Tensor):assert torch.equal(x,y)
            elif isinstance(x,dict):
                assert x.keys()==y.keys()
                for key in x:equal(x[key],y[key])
            elif isinstance(x,(tuple,list)):
                assert len(x)==len(y)
                for u,v in zip(x,y):equal(u,v)
            else:assert x==y
        for key in ('modelState','optimizerState','optimizerUpdates','rngState','objectiveState','scheduler'):equal(left[key],right[key])
        frozen=[key for key in left['modelState'] if key.endswith('.bias')]
        assert frozen and all(torch.equal(left['modelState'][key],previous['modelState'][key]) for key in frozen)
        weights=[key for key in left['modelState'] if key.endswith('.weight')]
        assert any(not torch.equal(left['modelState'][key],previous['modelState'][key]) for key in weights)
        for name in ('metrics.json','rate-trace.json'):assert json.loads((a/name).read_text())==json.loads((b/name).read_text())
        spec=importlib.util.spec_from_file_location('portable_inference',root/'portable'/'inference.py')
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        result=helper.load_model().predict(torch.tensor([[.25,-.5],[1.,2.]],dtype=torch.float32))
        assert list(result.shape)==[2,2] and torch.isfinite(result).all()
        print('inline-all-six-exact-resume-and-export-ok')
        """#.write(to:script,atomically:true,encoding:.utf8)
        let checked=expectation(description:"Exact CPU state and portable inference parity");var exitCode:Int32?,lines:[String]=[]
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:repository,allowDevelopment:true)
        let process=try WorkflowExecutionSession.launchWorker(runtime.executableURL,["-I",script.path,resumed.directory.appendingPathComponent("training/artifacts").path,full.directory.appendingPathComponent("training/artifacts").path,parent.directory.appendingPathComponent("training/artifacts").path],root,{lines.append($0)},{code,message in exitCode=code;if !message.isEmpty {lines.append(message)};checked.fulfill()})
        defer {process.cancel()}
        await fulfillment(of:[checked],timeout:30)
        XCTAssertEqual(exitCode,0,lines.joined(separator:"\n"))
        XCTAssertTrue(lines.contains("inline-all-six-exact-resume-and-export-ok"))
    }
}
