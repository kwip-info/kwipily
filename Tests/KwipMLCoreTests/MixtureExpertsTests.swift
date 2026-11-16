import XCTest
@testable import KwipMLCore

final class MixtureExpertsTests:XCTestCase {
    func testActualPointwiseShapesParameterCountAndScopedPorts()throws {
        let registry=try ComponentRegistry.builtIn(),graph=try MixtureExpertsPractice.graph(),node=graph.nodes[1]
        let c=try MixtureExpertsConfiguration(values:node.config,registry:registry)
        XCTAssertEqual(c.experts.count,4);XCTAssertEqual(c.parameterCount,180);XCTAssertEqual(c.trainableCount,180)
        XCTAssertTrue(GraphValidator.validate(graph,registry:registry).isValid)
        XCTAssertEqual(try c.outputLayout(.init(sample:[7,2])).sample.dimensions,[7,2])
        XCTAssertThrowsError(try c.outputLayout(.init(sample:[3])))
        XCTAssertEqual(registry.component(type:"mixture_experts")?.configuredInputs(node.config),["in"])
        var values=node.config;values["mask"]=true
        XCTAssertEqual(registry.component(type:"mixture_experts")?.configuredInputs(values),["in","mask"])
    }
    func testStrictIdentityAndGraphRefusals()throws {
        let registry=try ComponentRegistry.builtIn(),source=try MixtureExpertsPractice.graph().nodes[1].config
        var bad=source;bad["expert_2"]=source["expert_1"]
        XCTAssertThrowsError(try MixtureExpertsConfiguration(values:bad,registry:registry))
        bad=source;bad["top_k"]=4;bad["expert_count"]=2
        XCTAssertThrowsError(try MixtureExpertsConfiguration(values:bad,registry:registry))
        bad=source;bad["version"]=2
        XCTAssertThrowsError(try MixtureExpertsConfiguration(values:bad,registry:registry))
        let graph=try MixtureExpertsPractice.graph(),data=try JSONEncoder().encode(graph)
        XCTAssertEqual(try JSONDecoder().decode(GraphDocument.self,from:data),graph)
    }
    func testExpertLabelsAndLayoutAreCosmetic()throws {
        let original=try MixtureExpertsPractice.graph().nodes[1].config
        var updated=original
        var expert=try JSONDecoder().decode([String:JSONValue].self,from:Data(original["expert_1"]!.stringValue!.utf8))
        expert["name"] = .string("Renamed expert")
        guard case .object(var graph)=expert["graph"],case .array(var nodes)=graph["nodes"],case .object(var first)=nodes[0] else{return XCTFail("Fixture")}
        first["x"]=900;first["name"] = .string("Renamed input");nodes[0] = .object(first);graph["nodes"] = .array(nodes);expert["graph"] = .object(graph)
        updated["expert_1"] = .string(String(data:try JSONEncoder().encode(expert),encoding:.utf8)!)
        XCTAssertEqual(MixtureExpertsConfiguration.scientificValues(original),MixtureExpertsConfiguration.scientificValues(updated))
    }
    func testRoutingOutputsAndOldProjectRemainSeparate()throws {
        let registry=try ComponentRegistry.builtIn(),data=try DataFlowOperatorRegistry.load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json"))
        let document=try MixtureExpertsPractice.document()
        let train=try XCTUnwrap(document.nodes.first{$0.address==WorkflowExamples.trainAddress})
        let ports=try WorkflowPortContracts.resolve(train,document:document,modelRegistry:registry,dataRegistry:data)
        XCTAssertTrue(ports.outputs.contains{$0.id=="routing" && $0.kind == .table})
        let old=try WorkflowExamples.preparedParallelClassifier(),oldTrain=try XCTUnwrap(old.nodes.first{$0.address==WorkflowExamples.trainAddress})
        XCTAssertFalse(try WorkflowPortContracts.resolve(oldTrain,document:old,modelRegistry:registry,dataRegistry:data).outputs.contains{$0.id=="routing"})
    }
}
