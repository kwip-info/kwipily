import XCTest
import CryptoKit
import KwipMLCore
@testable import KwipMLApp

final class InitializationArtifactVerifierTests:XCTestCase {
    private func graph(id:String="layer",lazy:Bool=false)->GraphDocument {
        .init(nodes:[.init(id:"input",type:"input",config:["features":3]),.init(id:id,type:lazy ? "lazy_linear":"linear",config:lazy ? ["out_features":2,"bias":true]:["in_features":3,"out_features":2,"bias":true]),.init(id:"output",type:"output")],edges:[.init(source:"input",target:id),.init(source:id,target:"output")])
    }
    private func config(node:String="layer",parameter:String="weight",algorithm:String="zeros",settings:[String:JSONValue]=[:])throws->InitializationConfiguration {
        try .init(values:["seed":42,"rules":.array([.object(["owner":.object(["node":.string(node),"parameter":.string(parameter)]),"algorithm":.string(algorithm),"settings":.object(settings)])])])
    }
    private func owner(node:String="layer",parameter:String="weight",name:String?=nil,shape:[Int]=[2,3])->JSONValue {
        .object(["node":.string(node),"parameter":.string(parameter),"name":.string(name ?? "layers."+node+"."+parameter),"shape":.array(shape.map { .integer(Int64($0)) }),"dtype":.string("float32")])
    }
    private func manifest(_ c:InitializationConfiguration,owners:[JSONValue],mode:WorkflowTrainingMode = .fresh)->[String:JSONValue] {
        let identity:[String:JSONValue]=["schemaVersion":1,"recipe":c.recipeValue,"resolvedOwners":.array(owners),"generatorPlacement":.string("cpu-float32"),"seedDerivation":.string("kwip.initialization.owner-seed.v1")]
        var applied=identity;applied["torchVersion"] = .string("2.14.1");applied["application"] = .string(mode == .fresh ? "applied":"notAppliedOnRestore")
        return ["settings":.object(["initializationRecipe":c.recipeValue]),"runtime":.object(["torch":.string("2.14.1"),"device":.string("cpu")]),"provenance":.object(["initialization":.object(applied)]),"checkpoint":.object(["contract":.object(["initialization":.object(identity),"runtime":.object(["torch":.string("2.14.1")])])]),"lineage":.object(["mode":.string(mode == .fresh ? "newTraining":mode == .exactResume ? "exactResume":"weightsOnlyFineTune")])]
    }
    private func object(_ value:JSONValue?)throws->[String:JSONValue] {guard case .object(let o)=value else {throw WorkflowPhaseError("Expected test object")};return o}
    func testFreshAndRestorePoliciesRequireExactCanonicalMetadata()throws {
        let algorithms:[(String,[String:JSONValue])]=[("zeros",[:]),("normal",["mean":JSONValue.number(0.3),"std":.number(0.2)]),("xavier_uniform",["gain":.number(0.7)]),("kaiming_uniform",["a":.number(0.2),"mode":.string("fan_out"),"nonlinearity":.string("leaky_relu")])]
        for mode in WorkflowTrainingMode.allCases {
            for (algorithm,settings) in algorithms {
                let c=try config(algorithm:algorithm,settings:settings)
                XCTAssertNoThrow(try WorkflowInitializationVerifier.validate(manifest(c,owners:[owner()],mode:mode),expected:c,graph:graph(),trainingMode:mode))
            }
        }
    }
    func testAbsentRecipePreservesLegacyButRefusesEveryUnrequestedMetadataLocation()throws {
        let legacy:[String:JSONValue]=["settings":.object([:]),"provenance":.object([:]),"checkpoint":.object(["contract":.object([:])])]
        XCTAssertNoThrow(try WorkflowInitializationVerifier.validate(legacy,expected:nil,graph:graph(),trainingMode:.fresh))
        let c=try config(),full=manifest(c,owners:[owner()])
        for field in ["settings","provenance","checkpoint"] {
            var bad=legacy;bad[field]=full[field]
            XCTAssertThrowsError(try WorkflowInitializationVerifier.validate(bad,expected:nil,graph:graph(),trainingMode:.fresh),field)
        }
    }
    func testLegacyAbsenceDoesNotInventCheckpointContractRequirement()throws {
        let legacy:[String:JSONValue]=["settings":.object([:]),"provenance":.object([:])]
        XCTAssertNoThrow(try WorkflowInitializationVerifier.validate(legacy,expected:nil,graph:graph(),trainingMode:.fresh))
        var checkpointWithoutContract=legacy
        checkpointWithoutContract["checkpoint"] = .object(["schemaVersion":1,"path":.string("checkpoint.pt")])
        XCTAssertNoThrow(try WorkflowInitializationVerifier.validate(checkpointWithoutContract,expected:nil,graph:graph(),trainingMode:.fresh))
        let c=try config()
        XCTAssertThrowsError(try WorkflowInitializationVerifier.validate(legacy,expected:c,graph:graph(),trainingMode:.fresh))
        var unrequested=legacy
        unrequested["settings"] = .object(["initializationRecipe":c.recipeValue])
        XCTAssertThrowsError(try WorkflowInitializationVerifier.validate(unrequested,expected:nil,graph:graph(),trainingMode:.fresh))
    }
    func testOwnerShapeNameDtypeRecipeAndRuntimeTamperingRefuseEvenWhenRepinnedConsistently()throws {
        let c=try config(),valid=manifest(c,owners:[owner()])
        for mutation in ["shape","dtype","name","node","parameter","duplicate","empty","unknown","placement","seedVersion","torch","application","settings","missing"] {
            var bad=valid,checkpoint=try object(bad["checkpoint"]),contract=try object(checkpoint["contract"]),identity=try object(contract["initialization"]),provenance=try object(bad["provenance"])
            var selected=try object(owner())
            if mutation=="shape" {selected["shape"] = .array([3,2])}
            if mutation=="dtype" {selected["dtype"] = .string("float64")}
            if mutation=="name" {selected["name"] = .string("layers.foreign.weight")}
            if mutation=="node" {selected["node"] = .string("foreign")}
            if mutation=="parameter" {selected["parameter"] = .string("running_mean")}
            identity["resolvedOwners"] = .array([.object(selected)])
            if mutation=="duplicate" {identity["resolvedOwners"] = .array([owner(),owner()])}
            if mutation=="empty" {identity["resolvedOwners"] = .array([])}
            if mutation=="unknown" {identity["extra"] = true}
            if mutation=="placement" {identity["generatorPlacement"] = .string("mps")}
            if mutation=="seedVersion" {identity["seedDerivation"] = .string("unknown")}
            var applied=identity;applied["torchVersion"] = .string(mutation=="torch" ? "other":"2.14.1");applied["application"] = .string(mutation=="application" ? "notAppliedOnRestore":"applied")
            provenance["initialization"] = .object(applied);contract["initialization"] = .object(identity);checkpoint["contract"] = .object(contract)
            bad["checkpoint"] = .object(checkpoint);bad["provenance"] = .object(provenance)
            if mutation=="settings" {bad["settings"] = .object(["initializationRecipe":try config(parameter:"bias").recipeValue])}
            if mutation=="missing" {bad["provenance"] = .object([:])}
            XCTAssertThrowsError(try WorkflowInitializationVerifier.validate(bad,expected:c,graph:graph(),trainingMode:.fresh),mutation)
        }
        XCTAssertThrowsError(try WorkflowInitializationVerifier.validate(valid,expected:c,graph:graph(),trainingMode:.fineTune))
    }
    func testCollisionSafeKeysAndLazyMaterializedShapeAreIndependentlyPinned()throws {
        let digest=SHA256.hash(data:Data("items".utf8)).map {String(format:"%02x",$0)}.joined(),key="kwip_"+digest
        let c=try config(node:"items"),g=graph(id:"items",lazy:true)
        let valid=manifest(c,owners:[owner(node:"items",name:"layers."+key+".weight")])
        XCTAssertNoThrow(try WorkflowInitializationVerifier.validate(valid,expected:c,graph:g,trainingMode:.fresh))
        XCTAssertThrowsError(try WorkflowInitializationVerifier.validate(manifest(c,owners:[owner(node:"items")]),expected:c,graph:g,trainingMode:.fresh))
    }
    func testFrozenEmbeddingAndNestedRecurrentAttentionOwnersAreRealLearnedParameters()throws {
        var embedding=ExampleFlows.embeddingClassifier
        let i=try XCTUnwrap(embedding.nodes.firstIndex {$0.type == "embedding"});embedding.nodes[i].config["freeze"]=true
        let e=try config(node:"lookup")
        XCTAssertNoThrow(try WorkflowInitializationVerifier.validate(manifest(e,owners:[owner(node:"lookup",shape:[4,3])]),expected:e,graph:embedding,trainingMode:.fresh))
        let recurrent=try ExampleFlows.recurrentClassifier(componentID:"lstm"),r=try config(node:"memory",parameter:"weight_ih_l0")
        XCTAssertNoThrow(try WorkflowInitializationVerifier.validate(manifest(r,owners:[owner(node:"memory",parameter:"weight_ih_l0",shape:[8,1])]),expected:r,graph:recurrent,trainingMode:.fresh))
        let attention=try ExampleFlows.attentionClassifier(componentID:"multihead_attention")
        let node=try XCTUnwrap(attention.nodes.first {$0.type == "multihead_attention"}),a=try config(node:node.id,parameter:"out_proj.weight")
        XCTAssertNoThrow(try WorkflowInitializationVerifier.validate(manifest(a,owners:[owner(node:node.id,parameter:"out_proj.weight",shape:[2,2])]),expected:a,graph:attention,trainingMode:.fresh))
        let buffer=try config(node:"lookup",parameter:"running_mean")
        XCTAssertThrowsError(try WorkflowInitializationVerifier.validate(manifest(buffer,owners:[owner(node:"lookup",parameter:"running_mean",shape:[4])]),expected:buffer,graph:embedding,trainingMode:.fresh))
    }
}
