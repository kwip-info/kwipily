import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LossCompletedArtifactTests: XCTestCase {
    private let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    private func run(_ args: [String], log: URL) throws -> String {
        let python = repo.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: python.path) else { throw XCTSkip("Actual local Python runtime is required") }
        FileManager.default.createFile(atPath: log.path, contents: Data())
        let output = try FileHandle(forWritingTo: log); defer { try? output.close() }
        let process = Process(); process.executableURL = python; process.currentDirectoryURL = repo
        process.arguments = [repo.appendingPathComponent("runner/main.py").path] + args
        process.standardOutput = output; process.standardError = output
        try process.run(); let deadline = Date().addingTimeInterval(40)
        while process.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.025) }
        if process.isRunning { process.terminate(); XCTFail("Actual worker exceeded its bounded test deadline") }
        process.waitUntilExit()
        let bytes = try Data(contentsOf: log); XCTAssertLessThan(bytes.count, 2000000)
        let text = String(decoding: bytes, as: UTF8.self)
        XCTAssertEqual(process.terminationStatus, 0, String(text.suffix(3000)))
        return text
    }
    private func fixture(_ task: LossLearningTask, at root: URL, document: WorkflowDocument? = nil,retention:Bool=true) throws -> WorkflowPhaseExpectation {
        let modelRegistry = try ComponentRegistry.builtIn()
        let dataRegistry = try DataFlowOperatorRegistry.load(from: repo.appendingPathComponent("runner/dataflow-operators.json"))
        let plan = try WorkflowCompiler.compile(document ?? WorkflowExamples.preparedLossObjective(task: task, device: "cpu"), train: WorkflowExamples.trainAddress, modelRegistry: modelRegistry, dataRegistry: dataRegistry, contractVersion: 12)
        let graph = root.appendingPathComponent("dataflow.json")
        try DataFlowCodec.encodeGraph(plan.dataflow).write(to: graph)
        let resource = try XCTUnwrap(BundledLossComponents.entries[task.componentID]?.learning?.sourceResource)
        let bindings = root.appendingPathComponent("bindings.json")
        try WorkflowPhaseArtifacts.encode(["observations":repo.appendingPathComponent("Sources/KwipMLApp/Resources/"+resource+".csv").path]).write(to: bindings)
        let output = root.appendingPathComponent("preparation/artifacts")
        try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        _ = try run(["--dataflow-capture-outputs",retention ? "true":"false","--dataflow-required-nodes",String(decoding:try WorkflowPhaseArtifacts.encode(Array(Set([plan.training.node,plan.validation.node])).sorted()),as:UTF8.self),"--execute-dataflow",graph.path,"--source-bindings",bindings.path,"--dataflow-cache-policy",plan.preparationCachePolicy.rawValue,"--output",output.path], log: root.appendingPathComponent("preparation.log"))
        return .init(plan: plan, modelRegistrySHA256: try WorkflowArchiveCodec.registryHash(modelRegistry), dataRegistrySHA256: try WorkflowArchiveCodec.registryHash(dataRegistry), compilerContractVersion: 12,capturedModelRegistry:modelRegistry)
    }
    func testActualHistoryOffObjectiveKeepsReplayEvidenceWithoutObservationFiles()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer {try? FileManager.default.removeItem(at:root)}
        let base=try fixture(.mse,at:root,retention:false),policy=TrainingSampleCapturePolicy.factory.applyingHistory(outputs:false,diagnostics:false,requiresPredictionTable:false)
        let expected=WorkflowPhaseExpectation(plan:base.plan,modelRegistrySHA256:base.modelRegistrySHA256,dataRegistrySHA256:base.dataRegistrySHA256,compilerContractVersion:12,sampleCapture:policy,capturedModelRegistry:base.capturedModelRegistry)
        let prepared=try WorkflowPhaseArtifacts.preparation(expectation:expected,root:root)
        let graph=root.appendingPathComponent("model.json"),output=root.appendingPathComponent("training/artifacts")
        try WorkflowPhaseArtifacts.encode(expected.plan.model.graph).write(to:graph);try FileManager.default.createDirectory(at:output.deletingLastPathComponent(),withIntermediateDirectories:true)
        _=try run(["--graph",graph.path,"--output",output.path,"--device","cpu","--epochs","3","--batch-size","32","--seed","42","--learning-rate","0.03","--sample-capture-json",String(decoding:WorkflowPhaseArtifacts.encode(policy),as:UTF8.self)]+WorkflowPhaseArtifacts.preparedArguments(prepared.selection),log:root.appendingPathComponent("training.log"))
        let trained=try WorkflowPhaseArtifacts.training(expectation:expected,root:root,preparation:prepared.selection)
        XCTAssertEqual(trained.metrics.count,3);XCTAssertNotNil(trained.objectiveEvaluation);XCTAssertNotNil(trained.objectiveModelCode)
        XCTAssertNil(trained.objectivePredictionPreview);XCTAssertNil(trained.activationPreviews)
        XCTAssertTrue(trained.artifactFiles.contains {$0.path=="training/artifacts/checkpoint.pt"})
        XCTAssertFalse(trained.artifactFiles.contains {$0.path=="training/artifacts/predictions.json"})
    }
    func testAllEightActualCompletedRunsRetainTheirObjectiveArtifactsAndHonestAccuracy() throws {
        for task in LossLearningTask.allCases {
            let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
            defer {try? FileManager.default.removeItem(at:root)}
            let expected=try fixture(task,at:root)
            let prepared=try WorkflowPhaseArtifacts.preparation(expectation:expected,root:root)
            let graph=root.appendingPathComponent("model.json")
            try WorkflowPhaseArtifacts.encode(expected.plan.model.graph).write(to:graph)
            let output=root.appendingPathComponent("training/artifacts")
            try FileManager.default.createDirectory(at:output.deletingLastPathComponent(),withIntermediateDirectories:true)
            _=try run(["--graph",graph.path,"--output",output.path,"--device","cpu","--epochs","3","--batch-size","32","--seed","42","--learning-rate","0.03"]+WorkflowPhaseArtifacts.preparedArguments(prepared.selection),log:root.appendingPathComponent("training.log"))
            do {
                let trained=try WorkflowPhaseArtifacts.training(expectation:expected,root:root,preparation:prepared.selection)
                XCTAssertEqual(trained.metrics.count,3)
                XCTAssertNotNil(trained.objectiveContext)
                XCTAssertEqual(trained.objectivePredictionPreview?.rows.count,16)
                XCTAssertEqual(trained.objectiveEvaluation?.componentID,task.componentID)
                XCTAssertNotNil(trained.objectiveModelCode)
                if [.crossEntropy,.nll].contains(task) {XCTAssertNotNil(trained.metrics.last?.accuracy)} else {XCTAssertNil(trained.metrics.last?.accuracy)}
                XCTAssertTrue(trained.artifactFiles.contains {$0.path == "training/artifacts/objective_recipe.py"})
                XCTAssertTrue(trained.artifactFiles.contains {$0.path == "training/artifacts/predictions.json"})
                XCTAssertEqual(trained.partitionEvidence?.validationCount,16)
                XCTAssertNil(trained.predictionPreview,"Explicit objectives cannot masquerade as legacy binary prediction rows")
            } catch {XCTFail("\(task.rawValue): \(error)")}
        }
    }
    func testActualCompletedObjectiveRefusesMissingCaptureAndRepinnedCorruption()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let expected=try fixture(.mse,at:root),prepared=try WorkflowPhaseArtifacts.preparation(expectation:expected,root:root)
        let graph=root.appendingPathComponent("model.json"),output=root.appendingPathComponent("training/artifacts")
        try WorkflowPhaseArtifacts.encode(expected.plan.model.graph).write(to:graph)
        try FileManager.default.createDirectory(at:output.deletingLastPathComponent(),withIntermediateDirectories:true)
        _=try run(["--graph",graph.path,"--output",output.path,"--device","cpu","--epochs","3","--batch-size","32","--seed","42","--learning-rate","0.03"]+WorkflowPhaseArtifacts.preparedArguments(prepared.selection),log:root.appendingPathComponent("training.log"))
        XCTAssertNotNil(try WorkflowPhaseArtifacts.training(expectation:expected,root:root,preparation:prepared.selection).objectiveEvaluation)
        let noRegistry=WorkflowPhaseExpectation(plan:expected.plan,modelRegistrySHA256:expected.modelRegistrySHA256,dataRegistrySHA256:expected.dataRegistrySHA256,compilerContractVersion:12)
        XCTAssertThrowsError(try WorkflowPhaseArtifacts.training(expectation:noRegistry,root:root,preparation:prepared.selection),"No fallback to today's registry for historical evidence")
        var changedRegistry=try XCTUnwrap(expected.capturedModelRegistry)
        var registryJSON=try JSONDecoder().decode([String:JSONValue].self,from:WorkflowPhaseArtifacts.encode(changedRegistry))
        registryJSON["schemaVersion"] = .integer(99)
        changedRegistry=try JSONDecoder().decode(ComponentRegistry.self,from:WorkflowPhaseArtifacts.encode(registryJSON))
        let wrongRegistry=WorkflowPhaseExpectation(plan:expected.plan,modelRegistrySHA256:expected.modelRegistrySHA256,dataRegistrySHA256:expected.dataRegistrySHA256,compilerContractVersion:12,capturedModelRegistry:changedRegistry)
        XCTAssertThrowsError(try WorkflowPhaseArtifacts.training(expectation:wrongRegistry,root:root,preparation:prepared.selection),"Captured metadata must match its semantic pin")
        let names=["manifest.json","metrics.json","predictions.json","objective_recipe.py","checkpoint.pt"]
        let original=try Dictionary(uniqueKeysWithValues:names.map {($0,try Data(contentsOf:output.appendingPathComponent($0)))})
        func restore()throws {for(name,bytes)in original {try bytes.write(to:output.appendingPathComponent(name))}}
        let initial=try JSONDecoder().decode([String:JSONValue].self,from:original["manifest.json"]!)
        for mutation in 0..<7 {
            try restore();var manifest=initial
            switch mutation {
            case 0:
                manifest=Self.changed(manifest,path:["evaluation","accuracy"],value:.number(0))
            case 1:
                manifest=Self.changed(manifest,path:["evaluation","objective","rawPreview","shape"],value:.array([.integer(1)]))
            case 2:
                guard case .array(var rows)=try JSONDecoder().decode(JSONValue.self,from:original["metrics.json"]!),case .object(let first)=rows[0] else {throw WorkflowPhaseError("Actual epoch fixture missing")}
                rows[0] = .object(Self.changed(first,path:["objective","componentID"],value:.string("loss_l1")))
                let bytes=try WorkflowPhaseArtifacts.encode(JSONValue.array(rows));try bytes.write(to:output.appendingPathComponent("metrics.json"))
                manifest=Self.changed(manifest,path:["checkpoint","metricsSHA256"],value:.string(WorkflowPhaseArtifacts.hash(bytes)))
            case 3:
                var value=try JSONDecoder().decode([String:JSONValue].self,from:original["predictions.json"]!)
                guard case .array(var rows)=value["predictionRows"],case .object(let first)=rows[0] else {throw WorkflowPhaseError("Actual prediction fixture missing")}
                rows[0] = .object(Self.changed(first,path:["targets","fake"],value:.array([.number(0)])));value["predictionRows"] = .array(rows)
                let bytes=try WorkflowPhaseArtifacts.encode(value);try bytes.write(to:output.appendingPathComponent("predictions.json"))
                manifest=Self.changed(manifest,path:["predictionPreview","sha256"],value:.string(WorkflowPhaseArtifacts.hash(bytes)))
                manifest=Self.changed(manifest,path:["predictionPreview","bytes"],value:.integer(Int64(bytes.count)))
            case 4:
                try Data(repeating:32,count:131073).write(to:output.appendingPathComponent("objective_recipe.py"))
            case 5:
                try FileManager.default.removeItem(at:output.appendingPathComponent("objective_recipe.py"))
            default:
                var bytes=original["checkpoint.pt"]!;bytes.append(0);try bytes.write(to:output.appendingPathComponent("checkpoint.pt"))
            }
            try WorkflowPhaseArtifacts.encode(manifest).write(to:output.appendingPathComponent("manifest.json"))
            XCTAssertThrowsError(try WorkflowPhaseArtifacts.training(expectation:expected,root:root,preparation:prepared.selection),"Actual objective corruption \(mutation)")
        }
        try restore()
        XCTAssertEqual(try WorkflowPhaseArtifacts.training(expectation:expected,root:root,preparation:prepared.selection).objectivePredictionPreview?.rows.count,16)
    }
    func testActualExplicitCapturePoliciesSurviveCompletedVerification()throws {
        for activation in [false,true] {for rows in [false,true] {
            let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
            defer {try? FileManager.default.removeItem(at:root)}
            let base=try fixture(.mse,at:root),policy=TrainingSampleCapturePolicy(activationValues:activation,predictionRows:rows)
            let expected=WorkflowPhaseExpectation(plan:base.plan,modelRegistrySHA256:base.modelRegistrySHA256,dataRegistrySHA256:base.dataRegistrySHA256,compilerContractVersion:12,sampleCapture:policy,capturedModelRegistry:base.capturedModelRegistry)
            let prepared=try WorkflowPhaseArtifacts.preparation(expectation:expected,root:root)
            let graph=root.appendingPathComponent("model.json"),output=root.appendingPathComponent("training/artifacts")
            try WorkflowPhaseArtifacts.encode(expected.plan.model.graph).write(to:graph)
            try FileManager.default.createDirectory(at:output.deletingLastPathComponent(),withIntermediateDirectories:true)
            _=try run(["--graph",graph.path,"--output",output.path,"--device","cpu","--epochs","3","--batch-size","32","--seed","42","--learning-rate","0.03","--sample-capture-json",String(decoding:WorkflowPhaseArtifacts.encode(policy),as:UTF8.self)]+WorkflowPhaseArtifacts.preparedArguments(prepared.selection),log:root.appendingPathComponent("training.log"))
            let trained=try WorkflowPhaseArtifacts.training(expectation:expected,root:root,preparation:prepared.selection)
            XCTAssertEqual(trained.objectiveContext?.explicitSampleCapture,true)
            XCTAssertEqual(trained.objectivePredictionPreview?.rows.count,rows ? 16:0)
            XCTAssertEqual(trained.objectivePredictionPreview?.captureStatus,rows ? .captured:.omittedByPolicy)
            XCTAssertEqual(trained.objectiveEvaluation?.rawPreview?.values.isEmpty,!activation)
            XCTAssertNil(trained.metrics.last?.accuracy)
        }}
    }
    func testActualSelectedObjectiveOptimizerAndSchedulerCompleteTogether()throws {
        var document=try WorkflowExamples.preparedLossObjective(task:.crossEntropy,device:"cpu")
        let optimizer=try OptimizerConfiguration(componentID:"optimizer_sgd",values:["lr":.number(0.03)])
        let scheduler=try SchedulerConfiguration(componentID:"scheduler_step_lr",values:[:])
        let index=try XCTUnwrap(document.nodes.firstIndex {$0.address==WorkflowExamples.trainAddress})
        guard case .train(var config)=document.nodes[index].payload else {throw WorkflowArchiveError("Missing Train")}
        config["optimizer"]=optimizer.recipeValue;config["scheduler"]=scheduler.recipeValue
        document.nodes[index].payload = .train(config:config)
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let expected=try fixture(.crossEntropy,at:root,document:document)
        let prepared=try WorkflowPhaseArtifacts.preparation(expectation:expected,root:root)
        let graph=root.appendingPathComponent("model.json"),output=root.appendingPathComponent("training/artifacts")
        try WorkflowPhaseArtifacts.encode(expected.plan.model.graph).write(to:graph)
        try FileManager.default.createDirectory(at:output.deletingLastPathComponent(),withIntermediateDirectories:true)
        _=try run(["--graph",graph.path,"--output",output.path,"--device","cpu","--epochs","3","--batch-size","32","--seed","42","--learning-rate","0.03","--optimizer-recipe-json",String(decoding:WorkflowPhaseArtifacts.encode(optimizer.recipeValue),as:UTF8.self),"--scheduler-recipe-json",String(decoding:WorkflowPhaseArtifacts.encode(scheduler.recipeValue),as:UTF8.self)]+WorkflowPhaseArtifacts.preparedArguments(prepared.selection),log:root.appendingPathComponent("training.log"))
        let trained=try WorkflowPhaseArtifacts.training(expectation:expected,root:root,preparation:prepared.selection)
        XCTAssertEqual(trained.objectiveEvaluation?.componentID,"loss_cross_entropy")
        XCTAssertEqual(trained.metrics.count,3)
        XCTAssertTrue(trained.artifactFiles.contains {$0.path == "training/artifacts/rate-trace.json"})
    }
    private static func changed(_ root:[String:JSONValue],path:[String],value:JSONValue)->[String:JSONValue] {
        var result=root
        if path.count == 1 {result[path[0]]=value}
        else if case .object(let child)=result[path[0]] {result[path[0]] = .object(changed(child,path:Array(path.dropFirst()),value:value))}
        return result
    }

}
