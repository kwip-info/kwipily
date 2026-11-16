import XCTest
import KwipMLCore
@testable import KwipMLApp

/// Synthetic containers test host routing and file bounds; actual Torch acceptance is separate.
final class InitializationPhaseArtifactTests:XCTestCase {
    private let graph=GraphDocument(nodes:[.init(id:"input",type:"input",config:["features":2]),.init(id:"dense",type:"linear",config:["in_features":2,"out_features":2,"bias":true]),.init(id:"output",type:"output")],edges:[.init(source:"input",target:"dense"),.init(source:"dense",target:"output")])
    private func configuration()throws->InitializationConfiguration {
        try .init(values:["seed":42,"rules":.array([.object(["owner":.object(["node":.string("dense"),"parameter":.string("weight")]),"algorithm":.string("normal"),"settings":.object(["mean":0,"std":0.2])])])])
    }
    private func fixture(_ root:URL,selected:Bool=true)throws {
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let checkpoint=Data("Synthetic host inventory; not Torch acceptance".utf8)
        let metrics=Data("[{\"epoch\":1,\"trainLoss\":0.5,\"validationLoss\":0.6,\"accuracy\":0.7}]".utf8)
        var settings:[String:JSONValue]=["optimizer":.string("Adam"),"learningRate":0.03,"epochs":1,"partitionMode":.string("generatedSplit"),"requestedDevice":.string("cpu"),"requestedBatchSize":2]
        var provenance:[String:JSONValue]=[:],contract:[String:JSONValue]=["runtime":.object(["torch":.string("2.14.1")])]
        if selected {
            let recipe=try configuration().recipeValue
            let owner:JSONValue = .object(["node":.string("dense"),"parameter":.string("weight"),"name":.string("layers.dense.weight"),"shape":.array([2,2]),"dtype":.string("float32")])
            let identity:[String:JSONValue]=["schemaVersion":1,"recipe":recipe,"resolvedOwners":.array([owner]),"generatorPlacement":.string("cpu-float32"),"seedDerivation":.string("kwip.initialization.owner-seed.v1")]
            var actual=identity;actual["torchVersion"] = .string("2.14.1");actual["application"] = .string("applied")
            settings["initializationRecipe"]=recipe;contract["initialization"] = .object(identity);provenance["initialization"] = .object(actual)
        }
        let manifest:[String:JSONValue]=["schemaVersion":1,"status":.string("completed"),"runID":.string(UUID().uuidString),"seed":42,"graph":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(graph)),"settings":.object(settings),"provenance":.object(provenance),"runtime":.object(["torch":.string("2.14.1"),"device":.string("cpu")]),"lineage":.object(["mode":.string("newTraining")]),"checkpoint":.object(["contract":.object(contract),"sha256":.string(WorkflowPhaseArtifacts.hash(checkpoint)),"bytes":.integer(Int64(checkpoint.count)),"completedEpoch":1,"metricsSHA256":.string(WorkflowPhaseArtifacts.hash(metrics))])]
        try JSONEncoder().encode(manifest).write(to:root.appendingPathComponent("manifest.json"))
        try JSONEncoder().encode(graph).write(to:root.appendingPathComponent("graph.json"))
        try checkpoint.write(to:root.appendingPathComponent("checkpoint.pt"));try metrics.write(to:root.appendingPathComponent("metrics.json"))
        try Data("# synthetic model\n".utf8).write(to:root.appendingPathComponent("model.py"))
    }
    private func verify(_ root:URL,selected:Bool=true)throws->WorkflowPhaseArtifacts.Trained {
        try WorkflowPhaseArtifacts.trainingArtifacts(graph:graph,settings:.init(device:"cpu",epochs:1,batchSize:2,learningRate:0.03),preparation:nil,compositeSource:nil,trainingMode:.fresh,requiresLifecycleEvidence:false,lifecycleParent:nil,readRoot:root,parts:[],artifactPrefix:"",initializationRecipe:selected ? try configuration():nil)
    }
    func testSelectedPolicyRequiresBoundedUTF8HelperAndPinsItsExactBytes()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        try fixture(root)
        XCTAssertThrowsError(try verify(root),"Missing selected helper must refuse")
        let helper=Data("# synthetic helper tests inventory, not numerical equivalence\n".utf8)
        try helper.write(to:root.appendingPathComponent("initialization_recipe.py"))
        let accepted=try verify(root)
        XCTAssertEqual(accepted.hashes["initialization_recipe.py"],WorkflowPhaseArtifacts.hash(helper))
        XCTAssertEqual(accepted.artifactFiles.first {$0.path=="initialization_recipe.py"}?.bytes,helper.count)
        for invalid in [Data(),Data([0xff]),Data(repeating:65,count:131073)] {
            try invalid.write(to:root.appendingPathComponent("initialization_recipe.py"));XCTAssertThrowsError(try verify(root))
        }
    }
    func testAbsentPolicyRejectsUnrequestedMetadataAndHelper()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        try fixture(root)
        XCTAssertThrowsError(try verify(root,selected:false))
        try fixture(root,selected:false)
        XCTAssertNoThrow(try verify(root,selected:false))
        try Data("# unexpected policy\n".utf8).write(to:root.appendingPathComponent("initialization_recipe.py"))
        XCTAssertThrowsError(try verify(root,selected:false))
    }
    func testRepinnedContradictoryApplicationIsNotAcceptedBySharedPhaseRoute()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        try fixture(root)
        try Data("# bounded helper\n".utf8).write(to:root.appendingPathComponent("initialization_recipe.py"))
        let path=root.appendingPathComponent("manifest.json")
        var raw=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:path))
        guard case .object(var provenance)=raw["provenance"],case .object(var policy)=provenance["initialization"] else {return XCTFail("Fixture policy")}
        policy["application"] = .string("notAppliedOnRestore");provenance["initialization"] = .object(policy);raw["provenance"] = .object(provenance)
        try JSONEncoder().encode(raw).write(to:path)
        XCTAssertThrowsError(try verify(root))
    }
}
