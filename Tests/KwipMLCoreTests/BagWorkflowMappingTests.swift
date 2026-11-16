import XCTest
@testable import KwipMLCore

final class BagWorkflowMappingTests: XCTestCase {
    private let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    private func registry()throws->ComponentRegistry {
        var registry=try ComponentRegistry.builtIn()
        // Independent descriptor fixtures retain the shared field gate: these are
        // real declared controls, not empty templates that bypass configuration validation.
        func choice(_ key:String,_ value:String,_ options:[String])->ComponentField {
            .init(key:key,title:key,kind:.choice,defaultValue:.string(value),min:nil,max:nil,
                  choices:options.map {.init(value:$0,title:$0)})
        }
        func flag(_ key:String)->ComponentField {
            .init(key:key,title:key,kind:.boolean,defaultValue:false,min:nil,max:nil)
        }
        let sourceFields:[ComponentField]=[
            choice("layout","fixed",["fixed","packed"]),
            .init(key:"maximum_items_per_bag",title:"Maximum items",defaultValue:4,min:1,max:128),
            flag("per_sample_weights"),flag("include_last_offset")]
        let tableFields:[ComponentField]=[
            .init(key:"num_embeddings",title:"Table rows",defaultValue:16,min:1,max:100_000_000),
            .init(key:"embedding_dim",title:"Dimensions",defaultValue:4,min:1,max:4096),
            choice("mode","mean",["mean","sum","max"]),
            .init(key:"padding_idx",title:"Padding",defaultValue:.null,min:-100_000_000,max:99_999_999,allowsNull:true),
            .init(key:"max_norm",title:"Maximum norm",kind:.number,defaultValue:.null,min:1.1754943508222875e-38,max:3.4028234663852886e38,allowsNull:true),
            .init(key:"norm_type",title:"Norm order",kind:.number,defaultValue:2,min:1.1754943508222875e-38,max:3.4028234663852886e38),
            flag("scale_grad_by_freq"),flag("sparse"),flag("include_last_offset"),flag("freeze"),
            choice("layout","fixed",["fixed","packed"]),flag("per_sample_weights")]
        registry.components.removeAll {["bag_input","embedding_bag"].contains($0.id)}
        registry.components += [
            .init(id:"bag_input",title:"Bag input",inputs:[],outputs:["indices","offsets","per_sample_weights"],fields:sourceFields),
            .init(id:"embedding_bag",title:"Embedding bag",inputs:["indices","offsets","per_sample_weights"],outputs:["out"],fields:tableFields)]
        return registry
    }
    private func object(_ value:JSONValue)throws->[String:JSONValue] {
        guard case .object(let result)=value else {throw GraphConfigurationError("Expected mapping fixture")};return result
    }
    private func address(_ id:String,_ domain:WorkflowDomain = .control)->WorkflowNodeAddress { .init(domain:domain,localID:id) }
    private func raw(packed:Bool=false,weighted:Bool=false)->JSONValue {
        var v:[String:JSONValue]=["schemaVersion":1,"indicesColumns":.array(["i0","i1"].map(JSONValue.string))]
        if packed {v["lengthColumn"] = .string("length")}
        if weighted {v["weightColumns"] = .array(["w0","w1"].map(JSONValue.string))}
        return .object(v)
    }
    private func source(packed:Bool=false,weighted:Bool=false)throws->BagSourceConfiguration {
        var v=BagSourceConfiguration.defaults;v["maximum_items_per_bag"]=2
        v["layout"] = .string(packed ? "packed":"fixed");v["per_sample_weights"] = .bool(weighted)
        v["include_last_offset"] = .bool(packed)
        return try .init(values:v)
    }
    private func fixture(packed:Bool=false,weighted:Bool=false,objective:Bool=false)throws->WorkflowDocument {
        var doc=try WorkflowExamples.preparedEmbeddingClassifier(device:"cpu")
        doc.schemaVersion=12;doc.definitions=[];doc.drafts=[]
        doc.nodes.removeAll {$0.address.domain == .model}
        doc.edges.removeAll {$0.from.address.domain == .model || $0.to.address.domain == .model}
        let source=try source(packed:packed,weighted:weighted)
        var table=EmbeddingBagConfiguration.defaults;table["num_embeddings"]=11;table["embedding_dim"]=2
        // JSON has one numeric token for 2; JSONValue decodes that token as integer.
        // Author the wire-canonical numeric representation so full-document equality
        // remains strict, including exact bag mapping, node and wire ordering.
        table["norm_type"] = .integer(2)
        table["layout"] = .string(source.layout);table["per_sample_weights"] = .bool(weighted)
        table["include_last_offset"] = .bool(packed);table["mode"] = .string(weighted ? "sum":"mean")
        doc.nodes += [.init(address:address("bags",.model),payload:.modelPrimitive(type:"bag_input",configurationVersion:2,config:source.values)),
                      .init(address:address("table",.model),payload:.modelPrimitive(type:"embedding_bag",configurationVersion:2,config:table)),
                      .init(address:address("output",.model),payload:.modelPrimitive(type:"output",configurationVersion:2,config:[:]))]
        func wire(_ from:String,_ out:String,_ to:String,_ input:String) {
            doc.edges.append(.init(id:from+out+to+input,from:.init(address:address(from,.model),port:out),to:.init(address:address(to,.model),port:input),kind:.tensor))
        }
        for port in source.outputPorts {wire("bags",port,"table",port)}
        wire("table","out","output","in")
        doc.edges.append(.init(id:"definition",from:.init(address:address("output",.model),port:"definition"),to:.init(address:address("model"),port:"source"),kind:.modelDefinition))
        var config:[String:JSONValue]=["bagInput":raw(packed:packed,weighted:weighted),"targetColumn":.string("label")]
        if objective {
            let target=LossTargetMapping(componentID:"loss_mse",inputColumns:["i0","i1"],roles:[.init(role:"target",columns:["y0","y1"],sampleShape:[2],dtype:.float32)])
            config=["bagInput":raw(packed:packed,weighted:weighted),"lossTargetMapping":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(target))]
            doc.nodes.append(.init(address:address("objective"),payload:.lossRecipe(type:"loss_mse",configurationVersion:1,config:[:])))
            doc.edges.append(.init(id:"objective",from:.init(address:address("objective"),port:"out"),to:.init(address:WorkflowExamples.trainAddress,port:"objective"),kind:.objectiveRecipe))
        }
        let index=try XCTUnwrap(doc.nodes.firstIndex {$0.address == address("mapping")})
        doc.nodes[index].payload = .trainingData(config:config)
        return doc
    }
    private func compile(_ doc:WorkflowDocument,version:Int=12)throws->WorkflowExecutionPlan {
        try WorkflowCompiler.compile(RetiredTrainingRecipeMigration.migrate(doc),train:WorkflowExamples.trainAddress,modelRegistry:registry(),dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")),contractVersion:version)
    }
    func testExactMappingRoundTripBothLayoutsAndWeightModes()throws {
        for packed in [false,true] {for weighted in [false,true] {
            let mapping=try BagInputMapping(value:raw(packed:packed,weighted:weighted))
            XCTAssertEqual(mapping.value,raw(packed:packed,weighted:weighted))
            XCTAssertEqual(mapping.schemaVersion,1)
            XCTAssertNoThrow(try mapping.validate(sourceConfiguration:source(packed:packed,weighted:weighted),targetColumn:"label"))
            XCTAssertEqual(try JSONDecoder().decode(BagInputMapping.self,from:JSONEncoder().encode(mapping)),mapping)
            XCTAssertThrowsError(try mapping.validate(sourceConfiguration:source(packed:!packed,weighted:weighted),targetColumn:"label"))
        }}
    }
    func testStrictMappingFieldsColumnGeometryAndAllRoleDisjointness()throws {
        for bad:JSONValue in [.null,.object([:]),.object(["schemaVersion":2,"indicesColumns":.array(["i0","i1"].map(JSONValue.string))]),.object(["schemaVersion":1,"indicesColumns":.array(["i0","i0"].map(JSONValue.string))]),.object(["schemaVersion":1,"indicesColumns":.array([.string("i0"),1])]),.object(["schemaVersion":1,"indicesColumns":.array([" ","i1"].map(JSONValue.string))]),.object(["schemaVersion":1,"indicesColumns":.array(["i0","i1"].map(JSONValue.string)),"foreign":true])] {
            XCTAssertThrowsError(try BagInputMapping(value:bad))
        }
        let mapping=try BagInputMapping(value:raw(packed:true,weighted:true))
        XCTAssertThrowsError(try mapping.validate(sourceConfiguration:source(packed:true,weighted:true),targetColumn:"w0"))
        var v=try object(raw(packed:true,weighted:true));v["lengthColumn"] = .string("i0")
        XCTAssertThrowsError(try BagInputMapping(value:.object(v)))
        v=try object(raw(weighted:true));v["weightColumns"] = .array(["i1","w1"].map(JSONValue.string))
        XCTAssertThrowsError(try BagInputMapping(value:.object(v)))
        let wrong=LossTargetMapping(componentID:"loss_mse",inputColumns:["i1","i0"],roles:[.init(role:"target",columns:["y"],sampleShape:[],dtype:.float32)])
        XCTAssertThrowsError(try mapping.validate(sourceConfiguration:source(packed:true,weighted:true),objectiveMapping:wrong))
        let overlapping=LossTargetMapping(componentID:"loss_mse",inputColumns:["i0","i1"],roles:[.init(role:"target",columns:["w0"],sampleShape:[],dtype:.float32)])
        XCTAssertThrowsError(try mapping.validate(sourceConfiguration:source(packed:true,weighted:true),objectiveMapping:overlapping))
        XCTAssertThrowsError(try mapping.validate(sourceConfiguration:source(packed:true,weighted:true),targetColumn:"label",objectiveMapping:wrong))
    }
    func testCompilerRetainsTypedMappingAndSourceRequirementWithoutDenseSample()throws {
        for packed in [false,true] {for weighted in [false,true] {
            let doc=try fixture(packed:packed,weighted:weighted),plan=try compile(doc)
            XCTAssertEqual(plan.bagInput?.value,raw(packed:packed,weighted:weighted));XCTAssertEqual(plan.featureColumns,["i0","i1"])
            XCTAssertEqual(plan.targetColumn,"label");XCTAssertNil(plan.lossRecipe)
            let requirement=try WorkflowMappingRequirements.bagInputRequirement(plan.model.graph,registry:registry())
            XCTAssertEqual(requirement.sourceNodeID,"bags");XCTAssertEqual(requirement.configuration,try source(packed:packed,weighted:weighted))
            XCTAssertEqual(requirement.numEmbeddings,11);XCTAssertEqual(requirement.embeddingDim,2);XCTAssertEqual(requirement.preparationMode,"sum")
            XCTAssertThrowsError(try WorkflowMappingRequirements.inputContract(plan.model.graph,registry:registry()))
            let resolved=try WorkflowMappingRequirements.resolve(doc,mapping:address("mapping"),modelRegistry:registry(),dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")))
            XCTAssertEqual(resolved.first?.bagRequirement,requirement);XCTAssertNil(resolved.first?.contract);XCTAssertNil(resolved.first?.problem)
            XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(doc)),doc)
        }}
    }
    func testObjectiveMappingKeepsRealOutputShapeAndExactInputOrder()throws {
        let doc=try fixture(packed:true,weighted:true,objective:true),plan=try compile(doc)
        XCTAssertNil(plan.targetColumn);XCTAssertEqual(plan.lossRecipe?.configuration.componentID,"loss_mse")
        XCTAssertEqual(plan.lossRecipe?.targetMapping.inputColumns,["i0","i1"]);XCTAssertNotNil(plan.bagInput)
        XCTAssertEqual(try WorkflowMappingRequirements.outputContract(plan.model.graph,registry:registry()).layout.sample.dimensions,[2])
        var wrong=doc;let i=try XCTUnwrap(wrong.nodes.firstIndex {$0.address == address("mapping")})
        if case .trainingData(var config)=wrong.nodes[i].payload,case .object(var target)=config["lossTargetMapping"] {
            target["inputColumns"] = .array(["i1","i0"].map(JSONValue.string));config["lossTargetMapping"] = .object(target);wrong.nodes[i].payload = .trainingData(config:config)
        }
        XCTAssertThrowsError(try compile(wrong))
    }
    func testVersionGatesExplicitMigrationAndLegacyPlanRemainDistinct()throws {
        let old=try WorkflowExamples.preparedEmbeddingClassifier(device:"cpu"),before=try WorkflowCodec.encode(old)
        let migrated=try WorkflowCodec.migrateToBagInput(old)
        XCTAssertEqual(migrated.schemaVersion,12);var restored=migrated;restored.schemaVersion=old.schemaVersion
        XCTAssertEqual(restored,old);XCTAssertEqual(try WorkflowCodec.encode(restored),before)
        XCTAssertEqual(try WorkflowCodec.migrateToBagInput(migrated),migrated)
        XCTAssertNil(try compile(old,version:2).bagInput);XCTAssertNil(try compile(migrated).bagInput)
        var bag=try fixture();bag.schemaVersion=11;XCTAssertThrowsError(try WorkflowCodec.encode(bag))
        XCTAssertThrowsError(try compile(fixture(),version:10))
        var malformed=try fixture();let i=try XCTUnwrap(malformed.nodes.firstIndex {$0.address == address("mapping")})
        if case .trainingData(var config)=malformed.nodes[i].payload {config["featureColumns"] = .array(["i0","i1"].map(JSONValue.string));malformed.nodes[i].payload = .trainingData(config:config)}
        XCTAssertThrowsError(try WorkflowCodec.encode(malformed))
    }
    func testCompatibleFanoutUsesAllConsumersConservativelyAndPreservesNativeConfigs()throws {
        var doc=try fixture();let output=try XCTUnwrap(doc.nodes.firstIndex {$0.address == address("output",.model)})
        doc.edges.removeAll {$0.to.address == address("output",.model) && $0.kind == .tensor}
        var table=EmbeddingBagConfiguration.defaults;table["num_embeddings"]=7;table["embedding_dim"]=3;table["mode"] = .string("max")
        doc.nodes.insert(.init(address:address("second",.model),payload:.modelPrimitive(type:"embedding_bag",configurationVersion:2,config:table)),at:output)
        doc.nodes.append(.init(address:address("merge",.model),payload:.modelPrimitive(type:"concat",configurationVersion:2,config:["dim":-1])))
        doc.nodes.append(.init(address:address("head",.model),payload:.modelPrimitive(type:"linear",configurationVersion:2,config:["in_features":5,"out_features":2,"bias":true])))
        func wire(_ from:String,_ out:String,_ to:String,_ port:String) {doc.edges.append(.init(id:from+out+to+port,from:.init(address:address(from,.model),port:out),to:.init(address:address(to,.model),port:port),kind:.tensor))}
        wire("bags","indices","second","indices");wire("table","out","merge","a");wire("second","out","merge","b");wire("merge","out","head","in");wire("head","out","output","in")
        let plan=try compile(doc),requirement=try WorkflowMappingRequirements.bagInputRequirement(plan.model.graph,registry:registry())
        XCTAssertEqual(requirement.numEmbeddings,7);XCTAssertEqual(requirement.embeddingDim,3)
        XCTAssertEqual(plan.model.graph.nodes.first {$0.id == "table"}?.config["num_embeddings"],11)
        XCTAssertEqual(plan.model.graph.nodes.first {$0.id == "second"}?.config["mode"],.string("max"))
    }
    func testSourceMappingMismatchAndObjectiveConnectorMismatchRefuse()throws {
        var doc=try fixture();let i=try XCTUnwrap(doc.nodes.firstIndex {$0.address == address("mapping")})
        doc.nodes[i].payload = .trainingData(config:["bagInput":raw(packed:true),"targetColumn":.string("label")]);XCTAssertThrowsError(try compile(doc))
        doc=try fixture(objective:true);doc.edges.removeAll {$0.kind == .objectiveRecipe};XCTAssertThrowsError(try compile(doc))
        doc=try fixture();let sourceIndex=try XCTUnwrap(doc.nodes.firstIndex {$0.address == address("bags",.model)})
        doc.nodes[sourceIndex].payload = .modelPrimitive(type:"index_input",configurationVersion:2,config:["shape":.array([2])]);XCTAssertThrowsError(try compile(doc))
    }
    func testPendingOriginalCompositionDraftSurvivesBagMigrationWithoutApplying()throws {
        var source=try WorkflowExamples.preparedParallelClassifier(device:"cpu");source.schemaVersion=11
        let definition=try XCTUnwrap(source.definitions.first)
        let instance=try XCTUnwrap(source.nodes.first {if case .modelBlock = $0.payload {return true};return false})
        var proposed=definition;proposed.title="Unfinished container";proposed.body.edges=[]
        source.drafts=[.init(projectID:source.projectID,instancePath:[instance.address.localID],baseReference:definition.reference,baseRevision:source.revision,proposed:proposed,migrationOrigin:.workflow(schemaVersion:10))]
        let before=try WorkflowCodec.encode(source),migrated=try WorkflowCodec.migrateToBagInput(source)
        XCTAssertEqual(migrated.schemaVersion,12);XCTAssertEqual(migrated.drafts,source.drafts)
        XCTAssertEqual(migrated.projectID,source.projectID);XCTAssertEqual(migrated.revision,source.revision)
        XCTAssertEqual(migrated.definitions,source.definitions)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(migrated)),migrated)
        var restored=migrated;restored.schemaVersion=11;XCTAssertEqual(try WorkflowCodec.encode(restored),before)
        XCTAssertThrowsError(try compile(migrated))
        XCTAssertThrowsError(try CompositeDraftMigrationOrigin.workflow(schemaVersion:11).validate(workflowSchemaVersion:12))
    }
    func testBagExperimentVariantsKeepExactMappingAndCompilerEleven()throws {
        let source=try fixture(packed:true,weighted:true)
        let plan=try WorkflowExperimentPlan.build(source,train:WorkflowExamples.trainAddress,
            axes:[.init(field:"seed",values:[17,23])],budget:.init(maximumRuns:2,maximumTotalEpochs:6),
            modelRegistry:registry(),dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertEqual(plan.variants.count,2)
        XCTAssertEqual(plan.variants.map { $0.execution.bagInput?.value },[raw(packed:true,weighted:true),raw(packed:true,weighted:true)])
        XCTAssertEqual(plan.variants.map { $0.document.schemaVersion },[12,12])
    }

}
