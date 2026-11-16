import XCTest
@testable import KwipMLCore

final class GELUContractTests:XCTestCase {
    private func graph(_ approximate:Bool=false)->GraphDocument {
        GraphDocument(nodes:[GraphNode(id:"i",type:"input",config:["features":2]),GraphNode(id:"a",type:"linear",config:["in_features":2,"out_features":8]),GraphNode(id:"g",type:"gelu",config:["approximate_tanh":.bool(approximate)]),GraphNode(id:"b",type:"linear",config:["in_features":8,"out_features":2]),GraphNode(id:"o",type:"output")],edges:[GraphEdge(source:"i",target:"a"),GraphEdge(source:"a",target:"g"),GraphEdge(source:"g",target:"b"),GraphEdge(source:"b",target:"o")])
    }
    func testRegisteredBooleanModesPreserveShapeAndRejectMalformedConfiguration()throws {
        let registry=try ComponentRegistry.builtIn(),definition=try XCTUnwrap(registry.component(type:"gelu"))
        XCTAssertEqual(definition.inputs,["in"]);XCTAssertEqual(definition.outputs,["out"])
        XCTAssertEqual(definition.fields.map(\.key),["approximate_tanh"])
        XCTAssertEqual(definition.fields.first?.kind,.boolean);XCTAssertEqual(definition.fields.first?.defaultValue,.bool(false))
        for mode in [false,true] {let value=graph(mode),validation=GraphValidator.validate(value,registry:registry);XCTAssertTrue(validation.isValid,validation.issues.map(\.message).joined(separator:"; "));XCTAssertEqual(validation.widths["g"],8);XCTAssertEqual(try JSONDecoder().decode(GraphDocument.self,from:JSONEncoder().encode(value)),value)}
        for config:[String:JSONValue] in [[:],["approximate_tanh":1],["approximate_tanh":.string("true")],["approximate_tanh":.bool(false),"extra":.bool(true)]] {
            var bad=graph();bad.nodes[2].config=config
            XCTAssertFalse(GraphValidator.validate(bad,registry:registry).isValid)
            XCTAssertThrowsError(try GraphMigration.toCurrent(bad,registry:registry))
        }
        var mismatch=graph();mismatch.nodes[3].config["in_features"]=7;XCTAssertFalse(GraphValidator.validate(mismatch,registry:registry).isValid)
    }
    func testSchemaTwoOnlyAndModeChangesSemanticIdentityWithoutChangingLegacyBytes()throws {
        let registry=try ComponentRegistry.builtIn()
        XCTAssertNotEqual(try GraphSemantics.canonicalData(graph(),registry:registry),try GraphSemantics.canonicalData(graph(true),registry:registry))
        var historical=graph();historical.schemaVersion=1;for i in historical.nodes.indices where historical.nodes[i].type == "linear" {historical.nodes[i].config.removeValue(forKey:"bias")}
        XCTAssertThrowsError(try GraphMigration.toCurrent(historical,registry:registry));XCTAssertThrowsError(try GraphSemantics.canonicalData(historical,registry:registry))
        var previousRegistry=registry;previousRegistry.components.removeAll{$0.id == "gelu"}
        let previous=ExampleFlows.parallelClassifier
        XCTAssertEqual(try GraphSemantics.canonicalData(previous,registry:registry),try GraphSemantics.canonicalData(previous,registry:previousRegistry))
        var legacy=previous;legacy.schemaVersion=1;for i in legacy.nodes.indices where legacy.nodes[i].type == "linear" {legacy.nodes[i].config.removeValue(forKey:"bias")}
        XCTAssertEqual(try GraphSemantics.canonicalData(legacy,registry:registry),try GraphSemantics.canonicalData(previous,registry:registry))
    }
    func testCapabilityExplicitVersionFieldContractAndExistingLessonLinks()throws {
        let registry=try ComponentRegistry.builtIn(),package=try BuiltInCapabilityBridge.package(registry:registry,runtimeVersion:"2.14.1")
        let gelu=try XCTUnwrap(package.capabilities.first{$0.id == "gelu"})
        XCTAssertEqual(gelu.version,ContractVersion(1,1,0));XCTAssertEqual(gelu.adapterID,"kwip.runner.graph-v2.gelu");XCTAssertEqual(gelu.shapeRuleID,"kwip.shape.gelu.v1")
        XCTAssertTrue(gelu.lifecycle.state.isEmpty);XCTAssertEqual(gelu.learning.lessonIDs,[ModelConstructionLessonIdentity.geluExact.assessmentID,ModelConstructionLessonIdentity.geluTanh.assessmentID]);XCTAssertEqual(gelu.learning.exampleIDs,["kwip.example.gelu-exact-classifier.v1","kwip.example.gelu-tanh-classifier.v1"])
        XCTAssertEqual(gelu.ports.map{$0.schema.tensor?.dimensions},[[],[]])
        XCTAssertEqual(gelu.ports.map{$0.schema.tensor?.leadingAxes},[.init(minimum:2,maximum:8),.init(minimum:2,maximum:8)])
        XCTAssertTrue(BuiltInCapabilityBridge.validate(graph:graph(),registry:registry,runtimeVersion:"2.14.1").isValid)
        var bad=registry;let index=try XCTUnwrap(bad.components.firstIndex{$0.id == "gelu"});bad.components[index].fields[0].defaultValue = .bool(true)
        XCTAssertThrowsError(try BuiltInCapabilityBridge.declarations(registry:bad))
    }
    func testCompositeBooleanOverrideAndHistoricalLessonPoliciesRemainExplicit()throws {
        var project=CompositeExamples.parallelClassifier
        let index=try XCTUnwrap(project.definitions[0].body.nodes.firstIndex{$0.type == "relu"}),id=project.definitions[0].body.nodes[index].id
        project.definitions[0].body.nodes[index].type="gelu";project.definitions[0].body.nodes[index].config=["approximate_tanh":.bool(false)]
        let field=try XCTUnwrap(try ComponentRegistry.builtIn().component(type:"gelu")?.fields.first)
        project.definitions[0].parameters.append(CompositeParameter(field,targets:[CompositeParameterTarget(id,"approximate_tanh")]))
        project.root.instances[0].overrides["approximate_tanh"] = .bool(true)
        let expanded=try CompositeExpander.expand(project,registry:.builtIn())
        XCTAssertEqual(expanded.graph.nodes.first{$0.type == "gelu"}?.config["approximate_tanh"],.bool(true))
        XCTAssertEqual(project.definitions[0].body.nodes[index].config["approximate_tanh"],.bool(false))
        project.root.instances[0].overrides["approximate_tanh"]=1;XCTAssertThrowsError(try CompositeExpander.expand(project,registry:.builtIn()))
        for identity in [AffineLessonIdentity.composition,.backwardChain,.recovery(taskSeed:42)] {XCTAssertFalse(try AffineLessonTasks.publicTask(identity:identity).allowedNodeTypes.contains("gelu"))}
    }
    func testRuntimeDescriptorRequiresExactReviewedModesVersionAndKeys()throws {
        let valid:[String:Any]=["version":["major":1,"minor":0,"patch":0],"adapterID":"kwip.runner.graph-v2.gelu","graphSchemaVersions":[2],"approximateTanhModes":[false,true]]
        let value=try GeluRuntimeCapability.decode(JSONSerialization.data(withJSONObject:valid))
        XCTAssertEqual(value.version,ContractVersion());XCTAssertEqual(value.approximateTanhModes,[false,true])
        for (key,replacement) in [("version",["major":1,"minor":1,"patch":0] as Any),("adapterID","other" as Any),("graphSchemaVersions",[1,2] as Any),("approximateTanhModes",[false] as Any),("extra",true as Any)] {
            var bad=valid;bad[key]=replacement;XCTAssertThrowsError(try GeluRuntimeCapability.decode(JSONSerialization.data(withJSONObject:bad)),key)
        }
        var bad=valid;bad["version"]=["major":1,"minor":0,"patch":0,"extra":0];XCTAssertThrowsError(try GeluRuntimeCapability.decode(JSONSerialization.data(withJSONObject:bad)))
        bad=valid;bad["approximateTanhModes"]=[0,1];XCTAssertThrowsError(try GeluRuntimeCapability.decode(JSONSerialization.data(withJSONObject:bad)))
        XCTAssertThrowsError(try GeluRuntimeCapability.decode(Data(repeating:32,count:4097)))
    }

}
