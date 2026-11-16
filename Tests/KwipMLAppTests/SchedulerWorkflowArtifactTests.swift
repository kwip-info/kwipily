import XCTest
import KwipMLCore
@testable import KwipMLApp

final class SchedulerWorkflowArtifactTests:XCTestCase {
    private let graph:GraphDocument = .init(nodes:[.init(id:"input",type:"input",config:["features":2]),.init(id:"project",type:"linear",config:["in_features":2,"out_features":2]),.init(id:"output",type:"output")],edges:[.init(source:"input",target:"project"),.init(source:"project",target:"output")])
    private func object(_ value:JSONValue?)throws->[String:JSONValue] {guard case .object(let result)=value else {throw WorkflowPhaseError("Expected fixture object")};return result}
    private func optimizer()throws->OptimizerRecipe {.init(configuration:try .init(componentID:"optimizer_sgd",values:["lr":0.1,"momentum":0.5]),source:.init(domain:.control,localID:"adjustments"))}
    private let ownership:JSONValue = .array([.object(["name":.string("layers.project.weight"),"shape":.array([2,2]),"dtype":.string("float32")]),.object(["name":.string("layers.project.bias"),"shape":.array([2]),"dtype":.string("float32")])])
    private func recipe(_ id:String)throws->SchedulerRecipe {.init(configuration:try .init(componentID:id,values:[:]),source:.init(domain:.control,localID:"schedule"))}
    private func fixture(_ root:URL,id:String)throws->[String:JSONValue] {
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let opt=try optimizer(),schedule=try recipe(id),timing=schedule.configuration.timing
        let checkpoint=Data("Synthetic immutable bytes test native metadata admission, not Torch state".utf8)
        let metrics=Data("[{\"event\":\"epoch\",\"epoch\":1,\"trainLoss\":0.5,\"validationLoss\":0.6,\"accuracy\":0.7}]".utf8)
        let config=schedule.configuration.recipeValue
        let record:[String:JSONValue]
        var initialization:[String:JSONValue]=["tick":0,"rate":0.1]
        switch id {
        case "scheduler_one_cycle_lr":
            initialization["rate"]=0.004;initialization["momentum"]=0.95
            record=["tick":1,"epoch":1,"batch":1,"updatedParameterIndices":.array([0,1]),"usedRate":0.004,"nextRate":0.052,"nextRateUnused":false,"usedMomentum":0.95,"nextMomentum":0.9]
        case "scheduler_linear_lr":initialization["rate"] = .number(0.1/3);record=["tick":1,"epoch":1,"usedRate":.number(0.1/3),"nextRate":.number(0.1*(1.0/3+(1.0-1.0/3)/5)),"nextRateUnused":false]
        case "scheduler_cosine_annealing_lr":record=["tick":1,"epoch":1,"usedRate":0.1,"nextRate":.number(0.1*(1+cos(.pi/10))/2),"nextRateUnused":false]
        case "scheduler_reduce_on_plateau":record=["tick":1,"epoch":1,"usedRate":0.1,"nextRate":0.1,"nextRateUnused":false,"metric":.string("validation_loss"),"metricValue":0.6,"changed":false]
        default:record=["tick":1,"epoch":1,"usedRate":0.1,"nextRate":0.1,"nextRateUnused":false]
        }
        let trace:JSONValue = .object(["schemaVersion":1,"recipe":config,"timing":.string(timing),"initialization":.object(initialization),"records":.array([.object(record)])])
        let traceBytes=try JSONEncoder().encode(trace)
        let names=["scheduler_step_lr":"StepLR","scheduler_cosine_annealing_lr":"CosineAnnealingLR","scheduler_reduce_on_plateau":"ReduceLROnPlateau","scheduler_one_cycle_lr":"OneCycleLR","scheduler_linear_lr":"LinearLR"]
        let value:[String:JSONValue]=["schemaVersion":1,"status":.string("completed"),"runID":.string(UUID().uuidString),"seed":42,"graph":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(graph)),
            "settings":.object(["optimizer":.string("SGD"),"optimizerRecipe":opt.configuration.recipeValue,"optimizerRateSource":.string("explicitRecipe"),"learningRate":0.1,"schedulerRecipe":config,"epochs":1,"partitionMode":.string("generatedSplit"),"requestedDevice":.string("cpu"),"requestedBatchSize":2,"batchesPerEpoch":1,"initialEpoch":0,"epochsExecuted":1]),
            "runtime":.object(["torch":.string("2.14.1"),"device":.string("cpu")]),
            "provenance":.object(["optimizer":.object(["recipe":opt.configuration.recipeValue,"parameterOwnership":ownership,"implementation":.string("SGD"),"torchVersion":.string("2.14.1")]),"scheduler":.object(["recipe":config,"implementation":.string(names[id]!),"torchVersion":.string("2.14.1")])]),
            "checkpoint":.object(["completedEpoch":1,"bytes":.integer(Int64(checkpoint.count)),"sha256":.string(WorkflowPhaseArtifacts.hash(checkpoint)),"metricsSHA256":.string(WorkflowPhaseArtifacts.hash(metrics)),"contract":.object(["optimizer":.object(["schemaVersion":1,"recipe":opt.configuration.recipeValue,"parameterOwnership":ownership]),"scheduler":.object(["schemaVersion":1,"recipe":config])])]),
            "lineage":.object(["mode":.string("newTraining"),"parentRunID":.null,"restoredState":.array([]),"resetState":.array(["parameters","optimizer","randomGenerator","epoch","metrics","scheduler","rateTrace"].map(JSONValue.string))]),
            "rateTrace":.object(["schemaVersion":1,"path":.string("rate-trace.json"),"bytes":.integer(Int64(traceBytes.count)),"sha256":.string(WorkflowPhaseArtifacts.hash(traceBytes)),"records":1,"timing":.string(timing)])]
        try JSONEncoder().encode(value).write(to:root.appendingPathComponent("manifest.json"));try JSONEncoder().encode(graph).write(to:root.appendingPathComponent("graph.json"));try checkpoint.write(to:root.appendingPathComponent("checkpoint.pt"));try metrics.write(to:root.appendingPathComponent("metrics.json"));try traceBytes.write(to:root.appendingPathComponent("rate-trace.json"));try Data("# synthetic model export\n".utf8).write(to:root.appendingPathComponent("model.py"));try Data("# synthetic recipe text exercises bounded inventory only\n".utf8).write(to:root.appendingPathComponent("training_recipe.py"))
        return value
    }
    private func verifyWithoutScheduler(_ root:URL)throws {
        _=try WorkflowPhaseArtifacts.trainingArtifacts(graph:graph,settings:.init(device:"cpu",epochs:1,batchSize:2,learningRate:0.1),preparation:nil,compositeSource:nil,trainingMode:.fresh,requiresLifecycleEvidence:false,lifecycleParent:nil,readRoot:root,parts:[],artifactPrefix:"",optimizerRecipe:try optimizer())
    }
    func testUnrequestedSchedulerMetadataAndTraceAreNotSilentlyAccepted()throws {
        for id in SchedulerConfiguration.componentIDs.sorted() {
            let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer {try? FileManager.default.removeItem(at:root)}
            _=try fixture(root,id:id)
            XCTAssertThrowsError(try verifyWithoutScheduler(root),id)
        }
    }
    private func verify(_ root:URL,id:String,lifecycle:Bool=false)throws->WorkflowPhaseArtifacts.Trained {
        try WorkflowPhaseArtifacts.trainingArtifacts(graph:graph,settings:.init(device:"cpu",epochs:1,batchSize:2,learningRate:0.1),preparation:nil,compositeSource:nil,trainingMode:.fresh,requiresLifecycleEvidence:lifecycle,lifecycleParent:nil,readRoot:root,parts:[],artifactPrefix:"",optimizerRecipe:try optimizer(),schedulerRecipe:try recipe(id))
    }
    private func writeManifest(_ manifest:[String:JSONValue],_ root:URL)throws {try JSONEncoder().encode(manifest).write(to:root.appendingPathComponent("manifest.json"))}
    private func mutateTrace(_ root:URL,manifest:inout [String:JSONValue],change:(inout [String:JSONValue])throws->Void)throws {
        var trace=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:root.appendingPathComponent("rate-trace.json")))
        try change(&trace)
        let bytes=try JSONEncoder().encode(trace);try bytes.write(to:root.appendingPathComponent("rate-trace.json"))
        var metadata=try object(manifest["rateTrace"]);metadata["bytes"] = .integer(Int64(bytes.count));metadata["sha256"] = .string(WorkflowPhaseArtifacts.hash(bytes));manifest["rateTrace"] = .object(metadata)
        try writeManifest(manifest,root)
    }
    func testFiveIndependentCurvesAreVerifiedAndInventoried()throws {
        for id in SchedulerConfiguration.componentIDs.sorted() {
            let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
            _=try fixture(root,id:id)
            let verified=try verify(root,id:id,lifecycle:true)
            let saved=try Data(contentsOf:root.appendingPathComponent("rate-trace.json"))
            XCTAssertEqual(verified.hashes["rate-trace.json"],WorkflowPhaseArtifacts.hash(saved))
            XCTAssertTrue(verified.artifactFiles.contains {$0.path == "rate-trace.json" && $0.bytes == saved.count})
        }
    }
    func testTraceContentCannotBeChangedEvenWithFreshHash()throws {
        let mutations:[String:(inout [String:JSONValue])->Void]=[
            "usedRate":{$0["usedRate"]=0.2},"nextRate":{$0["nextRate"]=0.3},"epoch":{$0["epoch"]=2},"tick":{$0["tick"]=2},"unused":{$0["nextRateUnused"]=true},"extra":{$0["state"] = .string("unrequested")}]
        for id in SchedulerConfiguration.componentIDs.sorted() {for (name,change) in mutations {
            let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
            var manifest=try fixture(root,id:id)
            try mutateTrace(root,manifest:&manifest) { trace in guard case .array(var rows)=trace["records"] else {throw WorkflowPhaseError("Fixture records")};var row=try self.object(rows[0]);change(&row);rows[0] = .object(row);trace["records"] = .array(rows)}
            XCTAssertThrowsError(try verify(root,id:id),id+name)
        }}
    }
    func testInitializerAndMetricObservationsAreIndependentOfPins()throws {
        for id in SchedulerConfiguration.componentIDs.sorted() {
            let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
            var manifest=try fixture(root,id:id)
            try mutateTrace(root,manifest:&manifest) {trace in var initial=try self.object(trace["initialization"]);initial["rate"]=0.123;trace["initialization"] = .object(initial)}
            XCTAssertThrowsError(try verify(root,id:id),id)
        }
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        let id="scheduler_reduce_on_plateau";var manifest=try fixture(root,id:id)
        try mutateTrace(root,manifest:&manifest) {trace in guard case .array(var rows)=trace["records"] else {throw WorkflowPhaseError("Fixture records")};var row=try self.object(rows[0]);row["metricValue"]=0.61;rows[0] = .object(row);trace["records"] = .array(rows)}
        XCTAssertThrowsError(try verify(root,id:id))
    }
    func testHashPathSizeCountAndMissingTraceAreRefused()throws {
        for field in ["sha256","bytes","path","records","timing"] {
            let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
            let id="scheduler_step_lr";var manifest=try fixture(root,id:id);var metadata=try object(manifest["rateTrace"])
            switch field {case "sha256":metadata[field] = .string(String(repeating:"a",count:64));case "bytes":metadata[field]=4_000_001;case "records":metadata[field]=2;case "path":metadata[field] = .string("../rate-trace.json");default:metadata[field] = .string("beforeUpdate")}
            manifest["rateTrace"] = .object(metadata);try writeManifest(manifest,root);XCTAssertThrowsError(try verify(root,id:id),field)
        }
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        _=try fixture(root,id:"scheduler_step_lr");try FileManager.default.removeItem(at:root.appendingPathComponent("rate-trace.json"));XCTAssertThrowsError(try verify(root,id:"scheduler_step_lr"))
    }
    func testQualifiedSourceAndRecipeProvenanceAreNotGrantedImplicitly()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        let id="scheduler_step_lr";var manifest=try fixture(root,id:id)
        let wrong=SchedulerRecipe(configuration:try recipe(id).configuration,source:.init(domain:.model,localID:"schedule"))
        XCTAssertThrowsError(try WorkflowPhaseArtifacts.trainingArtifacts(graph:graph,settings:.init(device:"cpu",epochs:1,batchSize:2,learningRate:0.1),preparation:nil,compositeSource:nil,trainingMode:.fresh,requiresLifecycleEvidence:false,lifecycleParent:nil,readRoot:root,parts:[],artifactPrefix:"",optimizerRecipe:try optimizer(),schedulerRecipe:wrong))
        var provenance=try object(manifest["provenance"]);var scheduler=try object(provenance["scheduler"]);scheduler["implementation"] = .string("CosineAnnealingLR");provenance["scheduler"] = .object(scheduler);manifest["provenance"] = .object(provenance);try writeManifest(manifest,root)
        XCTAssertThrowsError(try verify(root,id:id))
    }
    func testOneCycleUpdateOwnershipAndMomentumAreStrict()throws {
        for key in ["updatedParameterIndices","batch","usedMomentum","nextMomentum"] {
            let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
            let id="scheduler_one_cycle_lr";var manifest=try fixture(root,id:id)
            try mutateTrace(root,manifest:&manifest) {trace in guard case .array(var rows)=trace["records"] else {throw WorkflowPhaseError("Fixture records")};var row=try self.object(rows[0]);row[key] = key == "updatedParameterIndices" ? .array([0,2]):key == "batch" ? 2:0.5;rows[0] = .object(row);trace["records"] = .array(rows)}
            XCTAssertThrowsError(try verify(root,id:id),key)
        }
    }
    func testUnscheduledOptionalFileWithoutMetadataIsRefused()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        var manifest=try fixture(root,id:"scheduler_step_lr");var settings=try object(manifest["settings"]);settings.removeValue(forKey:"schedulerRecipe");manifest["settings"] = .object(settings)
        var provenance=try object(manifest["provenance"]);provenance.removeValue(forKey:"scheduler");manifest["provenance"] = .object(provenance)
        var checkpoint=try object(manifest["checkpoint"]);var contract=try object(checkpoint["contract"]);contract.removeValue(forKey:"scheduler");checkpoint["contract"] = .object(contract);manifest["checkpoint"] = .object(checkpoint);manifest.removeValue(forKey:"rateTrace");try writeManifest(manifest,root)
        XCTAssertThrowsError(try verifyWithoutScheduler(root))
        try FileManager.default.removeItem(at:root.appendingPathComponent("rate-trace.json"));XCTAssertNoThrow(try verifyWithoutScheduler(root))
    }
    func testLiveRecordsShareStrictStructuralAdmission()throws {
        let schedule=try recipe("scheduler_one_cycle_lr").configuration
        let valid:JSONValue = .object(["tick":1,"epoch":1,"batch":1,"updatedParameterIndices":.array([0,1]),"usedRate":0.004,"nextRate":0.052,"nextRateUnused":false,"usedMomentum":0.95,"nextMomentum":0.9])
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validateSchedulerRecord(valid,configuration:schedule,expectedTick:1,parameterCount:2))
        let badCases:[[String:JSONValue]] = [["updatedParameterIndices":.array([1,0])],["updatedParameterIndices":.array([0,0])],["updatedParameterIndices":.array([2])],["tick":true],["nextRate":-0.1],["epoch":0],["nextRateUnused":true],["nextMomentum":.number(.infinity)],["foreign":true]]
        for bad in badCases {
            var value=try object(valid);value.merge(bad) {_,new in new};XCTAssertThrowsError(try WorkflowLifecycleVerifier.validateSchedulerRecord(.object(value),configuration:schedule,expectedTick:1,parameterCount:2))
        }
    }
    func testSchedulerLifecycleResetAndRestorationAreExplicit()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        var manifest=try fixture(root,id:"scheduler_step_lr")
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validate(manifest,mode:.fresh,targetEpochs:1,parent:nil,graph:graph,hasScheduler:true))
        XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(manifest,mode:.fresh,targetEpochs:1,parent:nil,graph:graph))
        let parent=WorkflowLifecycleParentEvidence(workerRunID:UUID().uuidString,checkpointSHA256:String(repeating:"a",count:64),completedEpoch:1,hasScheduler:true)
        manifest["lineage"] = .object(["mode":.string("weightsOnlyFineTune"),"parentRunID":.string(parent.workerRunID),"parentCheckpointSHA256":.string(parent.checkpointSHA256),"parentCompletedEpoch":1,"guarantee":.string("weightsOnly"),"restoredState":.array([.string("parameters")]),"resetState":.array(["optimizer","randomGenerator","epoch","metrics","scheduler","rateTrace"].map(JSONValue.string))])
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validate(manifest,mode:.fineTune,targetEpochs:1,parent:parent,graph:graph))
        var lineage=try object(manifest["lineage"]);lineage["resetState"] = .array(["optimizer","randomGenerator","epoch","metrics"].map(JSONValue.string));manifest["lineage"] = .object(lineage)
        XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(manifest,mode:.fineTune,targetEpochs:1,parent:parent,graph:graph))
    }

    func testExactResumeCannotRestoreSchedulerFromUnscheduledParent()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        var manifest=try fixture(root,id:"scheduler_step_lr")
        let parent=WorkflowLifecycleParentEvidence(workerRunID:UUID().uuidString,checkpointSHA256:String(repeating:"a",count:64),completedEpoch:1)
        var settings=try object(manifest["settings"]);settings["initialEpoch"]=1;settings["epochsExecuted"]=1;manifest["settings"] = .object(settings)
        manifest["lineage"] = .object(["mode":.string("exactResume"),"parentRunID":.string(parent.workerRunID),"parentCheckpointSHA256":.string(parent.checkpointSHA256),"parentCompletedEpoch":1,"guarantee":.string("epochBoundaryExactCPU"),"restoredState":.array(["parameters","optimizer","randomGenerator","epoch","metrics","scheduler","rateTrace"].map(JSONValue.string)),"resetState":.array([])])
        XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(manifest,mode:.exactResume,targetEpochs:2,parent:parent,graph:graph,hasScheduler:true))
        let selectedParent=WorkflowLifecycleParentEvidence(workerRunID:parent.workerRunID,checkpointSHA256:parent.checkpointSHA256,completedEpoch:1,hasScheduler:true)
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validate(manifest,mode:.exactResume,targetEpochs:2,parent:selectedParent,graph:graph,hasScheduler:true))
    }

    func testNativeLinearMomentumEndpointRemainsAdmissible()throws {
        // Installed Torch CPU OneCycle, total8/pct.5/linear/base.3/max.9 produces
        // this representable endpoint at tick3; strict ClosedRange rejects it.
        let configuration=try SchedulerConfiguration(componentID:"scheduler_one_cycle_lr",values:["planned_total_batches":8,"pct_start":0.5,"anneal_strategy":.string("linear"),"base_momentum":0.3,"max_momentum":0.9])
        let record:JSONValue = .object(["tick":3,"epoch":1,"batch":3,"updatedParameterIndices":.array([0,1]),"usedRate":0.068,"nextRate":0.1,"nextRateUnused":false,"usedMomentum":0.5,"nextMomentum":0.29999999999999993])
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validateSchedulerRecord(record,configuration:configuration,expectedTick:3,parameterCount:2))
        var changed=try object(record);changed["nextMomentum"]=0.299
        XCTAssertThrowsError(try WorkflowLifecycleVerifier.validateSchedulerRecord(.object(changed),configuration:configuration,expectedTick:3,parameterCount:2))
    }

}
