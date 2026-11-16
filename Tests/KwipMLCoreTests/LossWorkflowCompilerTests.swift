import XCTest
@testable import KwipMLCore

final class LossWorkflowCompilerTests:XCTestCase {
    private let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    private func registry()throws->ComponentRegistry {
        var registry=try ComponentRegistry.builtIn()
        let value=try JSONDecoder().decode(JSONValue.self,from:Data(contentsOf:root.appendingPathComponent("components/losses/bundle.json")))
        guard case .object(let bundle)=value,case .array(let rows)=bundle["components"] else {throw GraphConfigurationError("Missing test-local loss definitions.")}
        for row in rows {
            guard case .object(let entry)=row,let definition=entry["definition"] else {throw GraphConfigurationError("Missing loss definition.")}
            let reviewed=try JSONDecoder().decode(ComponentDefinition.self,from:JSONEncoder().encode(definition))
            if let existing=registry.components.first(where:{$0.id == reviewed.id}) {XCTAssertEqual(existing,reviewed)}
            else {registry.components.append(reviewed)}
        }
        return registry
    }
    private func compile(_ doc:WorkflowDocument,version:Int=12)throws->WorkflowExecutionPlan {
        try WorkflowCompiler.compile(RetiredTrainingRecipeMigration.migrate(doc),train:WorkflowExamples.trainAddress,modelRegistry:registry(),dataRegistry:DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json")),contractVersion:version)
    }
    private func addr(_ id:String,_ domain:WorkflowDomain = .control)->WorkflowNodeAddress {.init(domain:domain,localID:id)}
    private func mapping(_ id:String)->LossTargetMapping {
        let inputs=id == "loss_triplet_margin" ? ["a","b","c","d","e","f"]:["a","b"]
        if id == "loss_triplet_margin" {return .init(componentID:id,inputColumns:inputs,roles:[],tripletSourceGroups:[["a","b"],["c","d"],["e","f"]])}
        if id == "loss_ctc" {return .init(componentID:id,inputColumns:inputs,roles:[.init(role:"target",columns:["label"],sampleShape:[1],dtype:.int64),.init(role:"input_lengths",columns:["inputLength"],sampleShape:[],dtype:.int64),.init(role:"target_lengths",columns:["targetLength"],sampleShape:[],dtype:.int64)])}
        let indexed=id == "loss_cross_entropy" || id == "loss_nll"
        return .init(componentID:id,inputColumns:inputs,roles:[.init(role:"target",columns:indexed ? ["label"]:["y0","y1","y2"],sampleShape:indexed ? []:[3],dtype:indexed ? .int64:.float32)])
    }
    private func fixture(_ id:String)throws->WorkflowDocument {
        var doc=try WorkflowExamples.preparedParallelClassifier(device:"cpu");doc.schemaVersion=8
        doc.nodes.removeAll {$0.address.domain == .model}
        doc.edges.removeAll {$0.from.address.domain == .model || $0.to.address.domain == .model}
        let inputWidth=id == "loss_triplet_margin" ? 6:2
        let width=(id == "loss_ctc" || id == "loss_triplet_margin") ? 6:3
        doc.nodes += [
            .init(address:addr("input",.model),payload:.modelPrimitive(type:"input",configurationVersion:2,config:["features":.integer(Int64(inputWidth))])),
            .init(address:addr("head",.model),payload:.modelPrimitive(type:"linear",configurationVersion:2,config:["in_features":.integer(Int64(inputWidth)),"out_features":.integer(Int64(width)),"bias":true])),
            .init(address:addr("output",.model),payload:.modelPrimitive(type:"output",configurationVersion:2,config:[:]))]
        func wire(_ source:String,_ target:String) {doc.edges.append(.init(id:source+"-"+target,from:.init(address:addr(source,.model),port:"out"),to:.init(address:addr(target,.model),port:"in"),kind:.tensor))}
        wire("input","head")
        var previous="head"
        if width == 6 {
            let shape:[JSONValue]=id == "loss_ctc" ? [2,3]:[3,2]
            doc.nodes.append(.init(address:addr("shape",.model),payload:.modelPrimitive(type:"reshape",configurationVersion:2,config:["shape":.array(shape)])))
            wire(previous,"shape");previous="shape"
        }
        if ["loss_nll","loss_kl_div","loss_ctc"].contains(id) {
            doc.nodes.append(.init(address:addr("logs",.model),payload:.modelPrimitive(type:"log_softmax",configurationVersion:2,config:["dim":-1])))
            wire(previous,"logs");previous="logs"
        }
        wire(previous,"output")
        doc.edges.append(.init(id:"output-definition",from:.init(address:addr("output",.model),port:"definition"),to:.init(address:addr("model"),port:"source"),kind:.modelDefinition))
        let target=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(mapping(id)))
        let index=try XCTUnwrap(doc.nodes.firstIndex {$0.address == addr("mapping")})
        doc.nodes[index].payload = .trainingData(config:["lossTargetMapping":target])
        doc.nodes.append(.init(address:addr("objective"),payload:.lossRecipe(type:id,configurationVersion:1,config:[:])))
        doc.edges.append(.init(id:"objective-train",from:.init(address:addr("objective"),port:"out"),to:.init(address:WorkflowExamples.trainAddress,port:"objective"),kind:.objectiveRecipe))
        return doc
    }
    func testAllEightActualObjectiveMappingsAndQualifiedPlans()throws {
        for id in LossConfiguration.componentIDs.sorted() {
            let doc=try fixture(id)
            let historical=try WorkflowCompiler.compileHistoricalEvidence(doc,train:WorkflowExamples.trainAddress,modelRegistry:registry(),dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")),contractVersion:7)
            XCTAssertEqual(historical.lossRecipe?.source,addr("objective"))
            let plan=try compile(doc)
            XCTAssertEqual(plan.lossRecipe?.configuration,historical.lossRecipe?.configuration)
            XCTAssertEqual(plan.lossRecipe?.targetMapping,historical.lossRecipe?.targetMapping)
            XCTAssertEqual(plan.lossRecipe?.configuration.componentID,id)
            XCTAssertEqual(plan.lossRecipe?.source,WorkflowExamples.trainAddress)
            XCTAssertEqual(plan.lossRecipe?.targetMapping,mapping(id))
            XCTAssertEqual(plan.featureColumns,mapping(id).inputColumns)
            XCTAssertNil(plan.targetColumn,"No invented scalar target for explicit objectives")
            XCTAssertEqual(plan.objectiveScalarization.rawValue,"mean")
            XCTAssertFalse(plan.model.graph.nodes.contains {$0.type.hasPrefix("loss_")})
            let value=try WorkflowCodec.decode(WorkflowCodec.encode(doc))
            XCTAssertEqual(value,doc)
            XCTAssertThrowsError(try compile(doc,version:6))
        }
    }
    func testActualOutputRolesAndShapesRefuseRatherThanCoercing()throws {
        var doc=try fixture("loss_mse")
        let i=try XCTUnwrap(doc.nodes.firstIndex {$0.address == addr("mapping")})
        var wrong=mapping("loss_mse");wrong.roles[0].sampleShape=[1];wrong.roles[0].columns=["y0"]
        doc.nodes[i].payload = .trainingData(config:["lossTargetMapping":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(wrong))])
        XCTAssertThrowsError(try compile(doc))
        doc=try fixture("loss_cross_entropy")
        let loss=try XCTUnwrap(doc.nodes.firstIndex {$0.address == addr("objective")})
        doc.nodes[loss].payload = .lossRecipe(type:"loss_cross_entropy",configurationVersion:1,config:["class_weights":.array([1,2])])
        XCTAssertThrowsError(try compile(doc))
        doc=try fixture("loss_ctc");let ctcLoss=try XCTUnwrap(doc.nodes.firstIndex {$0.address == addr("objective")});doc.nodes[ctcLoss].payload = .lossRecipe(type:"loss_ctc",configurationVersion:1,config:["blank":3])
        XCTAssertThrowsError(try compile(doc))
        doc=try fixture("loss_mse");wrong=mapping("loss_l1")
        let mi=try XCTUnwrap(doc.nodes.firstIndex {$0.address == addr("mapping")})
        doc.nodes[mi].payload = .trainingData(config:["lossTargetMapping":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(wrong))])
        XCTAssertThrowsError(try compile(doc))
    }
    func testLegacyAndDisconnectedOutputAdmissionRemainStrict()throws {
        let old=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let baseline=try compile(old,version:2)
        XCTAssertEqual(baseline.targetColumn,"label");XCTAssertNil(baseline.lossRecipe)
        var disconnected=old;disconnected.schemaVersion=8
        disconnected.nodes.append(.init(address:addr("objective"),payload:.lossRecipe(type:"loss_mse",configurationVersion:1,config:[:])))
        XCTAssertThrowsError(try compile(disconnected),"An orphan recipe requires explicit repair rather than being silently ignored in new execution.")
        XCTAssertNil(try WorkflowCompiler.compileHistoricalEvidence(disconnected,train:WorkflowExamples.trainAddress,modelRegistry:registry(),dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")),contractVersion:7).lossRecipe)
        let output=try fixture("loss_mse")
        XCTAssertThrowsError(try WorkflowCompiler.modelSnapshot(output,definition:addr("model"),modelRegistry:registry(),dataRegistry:DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))))
        var malformed=output;malformed.schemaVersion=7;XCTAssertThrowsError(try WorkflowCodec.encode(malformed))
        malformed=output;malformed.edges[malformed.edges.count-1].kind = .optimizerRecipe;XCTAssertThrowsError(try compile(malformed))
        malformed=output;malformed.edges.append(malformed.edges.last!);XCTAssertThrowsError(try compile(malformed))
        malformed=output;malformed.nodes.removeAll {$0.address == addr("objective")};XCTAssertThrowsError(try compile(malformed))
    }
    func testScalarizationIsTypedAndOldSchemasRefuseIt()throws {
        var doc=try fixture("loss_mse")
        let train=try XCTUnwrap(doc.nodes.firstIndex {$0.address == WorkflowExamples.trainAddress})
        doc.nodes[train].payload = .train(config:["objectiveScalarization":.string("sum")])
        XCTAssertEqual(try compile(doc).objectiveScalarization.rawValue,"sum")
        for value:JSONValue in [.bool(true),.number(1),.string("median")] {
            doc.nodes[train].payload = .train(config:["objectiveScalarization":value]);XCTAssertThrowsError(try compile(doc))
        }
        var old=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let oi=try XCTUnwrap(old.nodes.firstIndex {$0.address == WorkflowExamples.trainAddress})
        old.nodes[oi].payload = .train(config:["objectiveScalarization":.string("sum")]);XCTAssertThrowsError(try WorkflowCodec.encode(old))
    }
    func testObjectiveRoleRankLimitAndStrictMappingKeys()throws {
        let target=LossTargetMapping(componentID:"loss_mse",inputColumns:["a","b"],roles:[.init(role:"target",columns:["y"],sampleShape:Array(repeating:1,count:8),dtype:.float32)])
        let recipe=try LossRecipe(configuration:LossConfiguration(componentID:"loss_mse",values:["target_broadcast":true]),source:addr("objective"),targetMapping:target)
        XCTAssertThrowsError(try WorkflowMappingRequirements.validateObjective(prediction:.init(layout:BatchedTensorLayout(sample:[3]),dtype:.float32),recipe:recipe))
        var doc=try fixture("loss_mse")
        let index=try XCTUnwrap(doc.nodes.firstIndex {$0.address == addr("mapping")})
        let raw=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(mapping("loss_mse")))
        guard case .object(var fields)=raw else {return XCTFail("Missing typed mapping fixture")}
        fields["answer"] = .array([1])
        doc.nodes[index].payload = .trainingData(config:["lossTargetMapping":.object(fields)])
        XCTAssertThrowsError(try WorkflowCodec.encode(doc))
        doc=try fixture("loss_mse")
        let loss=try XCTUnwrap(doc.nodes.firstIndex {$0.address == addr("objective")})
        doc.nodes[loss].address = addr("objective",.model)
        XCTAssertThrowsError(try WorkflowCodec.encode(doc))
    }

}
