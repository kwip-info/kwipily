import XCTest
import KwipMLCore
@testable import KwipMLApp

final class OptimizerWorkflowArtifactTests:XCTestCase {
    private func object(_ value:JSONValue?)throws->[String:JSONValue] {
        guard case .object(let result)=value else {throw WorkflowPhaseError("Expected fixture object")}
        return result
    }
    private let graph:GraphDocument = .init(nodes:[
        .init(id:"input",type:"input",config:["features":2]),
        .init(id:"project",type:"linear",config:["in_features":2,"out_features":2]),
        .init(id:"output",type:"output")],edges:[.init(source:"input",target:"project"),.init(source:"project",target:"output")])
    private let ownership:JSONValue = .array([
        .object(["name":.string("layers.project.weight"),"shape":.array([2,2]),"dtype":.string("float32")]),
        .object(["name":.string("layers.project.bias"),"shape":.array([2]),"dtype":.string("float32")])])
    private func recipe(_ id:String)throws->OptimizerRecipe {
        .init(configuration:try .init(componentID:id,values:[:]),source:.init(domain:.control,localID:"adjustments"))
    }
    private func metadata(_ recipe:OptimizerRecipe)->[String:JSONValue] {
        let names=["optimizer_sgd":"SGD","optimizer_adam":"Adam","optimizer_adamw":"AdamW","optimizer_rmsprop":"RMSprop","optimizer_adagrad":"Adagrad"]
        let name=names[recipe.configuration.componentID]!
        return ["settings":.object(["optimizer":.string(name),"optimizerRecipe":recipe.configuration.recipeValue,"optimizerRateSource":.string("explicitRecipe"),"learningRate":.number(recipe.configuration.learningRate)]),
            "runtime":.object(["torch":.string("2.14.1"),"device":.string("cpu")]),
            "provenance":.object(["optimizer":.object(["recipe":recipe.configuration.recipeValue,"parameterOwnership":ownership,"implementation":.string(name),"torchVersion":.string("2.14.1")])]),
            "checkpoint":.object(["contract":.object(["optimizer":.object(["schemaVersion":1,"recipe":recipe.configuration.recipeValue,"parameterOwnership":ownership])])])]
    }
    func testFiveExactRecipesRequireCanonicalProvenanceAndIndependentGraphOwnership()throws {
        for id in OptimizerConfiguration.componentIDs.sorted() {
            let expected=try recipe(id)
            XCTAssertNoThrow(try WorkflowLifecycleVerifier.validateOptimizer(metadata(expected),expected:expected,graph:graph))
        }
    }
    func testRecipeRateProvenanceCounterAndOwnershipTamperingRefuse()throws {
        let expected=try recipe("optimizer_adamw"),valid=metadata(expected)
        for mutation in ["rate","marker","name","recipe","implementation","torch","extraState","duplicate","wrongShape","buffer","dtype","empty"] {
            var bad=valid
            if case .object(var settings)=bad["settings"] {
                if mutation=="rate" {settings["learningRate"]=0.7}
                if mutation=="marker" {settings["optimizerRateSource"] = .string("runSetup")}
                if mutation=="name" {settings["optimizer"] = .string("Adam")}
                if mutation=="recipe" {settings["optimizerRecipe"]=try recipe("optimizer_sgd").configuration.recipeValue}
                bad["settings"] = .object(settings)
            }
            if case .object(var provenance)=bad["provenance"],case .object(var optimizer)=provenance["optimizer"] {
                if mutation=="implementation" {optimizer["implementation"] = .string("Adam")}
                if mutation=="torch" {optimizer["torchVersion"] = .string("changed")}
                if mutation=="extraState" {optimizer["optimizerUpdates"] = .array([1,1])}
                if mutation=="empty" {optimizer["parameterOwnership"] = .array([])}
                if ["duplicate","wrongShape","buffer","dtype"].contains(mutation),case .array(var rows)=ownership {
                    if mutation=="duplicate" {rows[1]=rows[0]}
                    if case .object(var item)=rows[0] {
                        if mutation=="wrongShape" {item["shape"] = .array([2,3])}
                        if mutation=="buffer" {item["name"] = .string("layers.project.running_mean")}
                        if mutation=="dtype" {item["dtype"] = .string("int64")}
                        rows[0] = .object(item)
                    }
                    optimizer["parameterOwnership"] = .array(rows)
                }
                provenance["optimizer"] = .object(optimizer);bad["provenance"] = .object(provenance)
            }
            XCTAssertThrowsError(try WorkflowLifecycleVerifier.validateOptimizer(bad,expected:expected,graph:graph),mutation)
        }
        var checkpointBad=valid
        if case .object(var checkpoint)=checkpointBad["checkpoint"],case .object(var contract)=checkpoint["contract"],case .object(var optimizer)=contract["optimizer"] {
            optimizer["counterDType"] = .string("int64");contract["optimizer"] = .object(optimizer);checkpoint["contract"] = .object(contract);checkpointBad["checkpoint"] = .object(checkpoint)
        }
        XCTAssertThrowsError(try WorkflowLifecycleVerifier.validateOptimizer(checkpointBad,expected:expected,graph:graph))
    }
    func testLegacyAbsentRecipeRetainsAdamBoundaryAndRejectsExplicitMetadata()throws {
        let legacy:[String:JSONValue]=["settings":.object(["optimizer":.string("Adam"),"learningRate":0.03]),"provenance":.object([:])]
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validateOptimizer(legacy,expected:nil,graph:graph))
        XCTAssertThrowsError(try WorkflowLifecycleVerifier.validateOptimizer(metadata(try recipe("optimizer_adam")),expected:nil,graph:graph))
    }
    func testTrainingRecipeIsBoundedUTF8AndIncludedInImmutableInventory()throws {
        let expected=try recipe("optimizer_sgd")
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        var value=metadata(expected)
        let checkpoint=Data("Synthetic immutable bytes, not a Torch state acceptance".utf8)
        let metrics=Data("[{\"epoch\":1,\"trainLoss\":0.5,\"validationLoss\":0.6,\"accuracy\":0.7}]".utf8)
        var settings=try object(value["settings"])
        settings.merge(["epochs":1,"partitionMode":.string("generatedSplit"),"requestedDevice":.string("cpu"),"requestedBatchSize":2]) {_,v in v}
        value["settings"] = .object(settings);value["schemaVersion"]=1;value["status"] = .string("completed");value["runID"] = .string(UUID().uuidString);value["seed"]=42
        value["graph"]=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(graph))
        var metadata=try object(value["checkpoint"])
        metadata.merge(["sha256":.string(WorkflowPhaseArtifacts.hash(checkpoint)),"bytes":.integer(Int64(checkpoint.count)),"completedEpoch":1,"metricsSHA256":.string(WorkflowPhaseArtifacts.hash(metrics))]) {_,v in v}
        value["checkpoint"] = .object(metadata)
        try JSONEncoder().encode(value).write(to:root.appendingPathComponent("manifest.json"))
        try JSONEncoder().encode(graph).write(to:root.appendingPathComponent("graph.json"))
        try checkpoint.write(to:root.appendingPathComponent("checkpoint.pt"));try metrics.write(to:root.appendingPathComponent("metrics.json"))
        try Data("# model architecture\n".utf8).write(to:root.appendingPathComponent("model.py"))
        let recipeCode=Data("# synthetic text exercises bounded inventory, not executable recipe equivalence\n".utf8)
        try recipeCode.write(to:root.appendingPathComponent("training_recipe.py"))
        func verify()throws->WorkflowPhaseArtifacts.Trained {
            try WorkflowPhaseArtifacts.trainingArtifacts(graph:graph,settings:.init(device:"cpu",epochs:1,batchSize:2,learningRate:expected.configuration.learningRate),preparation:nil,compositeSource:nil,trainingMode:.fresh,requiresLifecycleEvidence:false,lifecycleParent:nil,readRoot:root,parts:[],artifactPrefix:"",optimizerRecipe:expected)
        }
        let accepted=try verify()
        XCTAssertEqual(accepted.hashes["training_recipe.py"],WorkflowPhaseArtifacts.hash(recipeCode))
        XCTAssertEqual(accepted.artifactFiles.first {$0.path=="training_recipe.py"}?.bytes,recipeCode.count)
        try Data(repeating:65,count:8193).write(to:root.appendingPathComponent("training_recipe.py"));XCTAssertThrowsError(try verify())
        try Data([0xff]).write(to:root.appendingPathComponent("training_recipe.py"));XCTAssertThrowsError(try verify())
    }
}
