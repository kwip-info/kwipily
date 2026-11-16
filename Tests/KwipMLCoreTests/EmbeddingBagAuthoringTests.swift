import XCTest
@testable import KwipMLCore
final class EmbeddingBagAuthoringTests:XCTestCase {
    func testPacketPortsCannotBeDiscardedInGraphOrWorkflow()throws {
        let registry=try ComponentRegistry.builtIn()
        var config=EmbeddingBagConfiguration.defaults
        config["layout"] = .string("packed");config["mode"] = .string("sum");config["per_sample_weights"]=true
        let lookup=GraphNode(id:"lookup",type:"embedding_bag",config:config,parameterSettings:["weight":.object(["trainability":.string("frozen")])])
        var source=BagSourceConfiguration.defaults;source["layout"] = .string("packed");source["per_sample_weights"]=true
        let edges=["indices","offsets","per_sample_weights"].map {GraphEdge(id:$0,source:"source",sourcePort:$0,target:"lookup",targetPort:$0)}
        let graph=GraphDocument(schemaVersion:2,nodes:[.init(id:"source",type:"bag_input",config:source),lookup],edges:edges)
        var invalid=config;invalid["layout"] = .string("fixed")
        XCTAssertThrowsError(try GraphCommandEngine.apply(.setEmbeddingBag(nodeID:"lookup",configuration:invalid),to:.init(document:graph),registry:registry))
        invalid=config;invalid["per_sample_weights"]=false
        XCTAssertThrowsError(try GraphCommandEngine.apply(.setEmbeddingBag(nodeID:"lookup",configuration:invalid),to:.init(document:graph),registry:registry))
        var valid=config;valid["padding_idx"] = -1
        let changed=try GraphCommandEngine.apply(.setEmbeddingBag(nodeID:"lookup",configuration:valid),to:.init(document:graph),registry:registry).after.document
        XCTAssertEqual(changed.edges,edges);XCTAssertEqual(changed.nodes.last?.parameterSettings,lookup.parameterSettings)
        XCTAssertEqual(changed.nodes.last?.config["padding_idx"],15)
        var workflow=try WorkflowCodec.migrateModel(.legacyFlat(graph),projectID:UUID().uuidString).document
        workflow.schemaVersion=max(11,workflow.schemaVersion)
        let address=WorkflowNodeAddress(domain:.model,localID:"lookup")
        XCTAssertThrowsError(try WorkflowEditing.apply(.setEmbeddingBag(address:address,configuration:invalid),to:workflow,modelRegistry:registry,dataRegistry:.load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json"))))
    }
    func testInlineBoundaryAndDimensionBindingRemainProtected()throws {
        let registry=try ComponentRegistry.builtIn()
        var config=EmbeddingBagConfiguration.defaults;config["layout"] = .string("packed")
        let field=try XCTUnwrap(registry.component(type:"embedding_bag")?.fields.first {$0.key=="embedding_dim"})
        let definition=CompositeDefinition(reference:.init("bag-container"),title:"Bag",body:.init(nodes:[.init(id:"lookup",type:"embedding_bag",config:config)],configurationVersion:2),inputs:[.init("indices",targets:[.init("lookup","indices")]),.init("offsets",targets:[.init("lookup","offsets")])],outputs:[.init("out",source:.init("lookup","out"))],parameters:[.init(field,targets:[.init("lookup","embedding_dim")])],composition:.init(kind:.graph,children:[.init(localID:"lookup")]))
        let root=WorkflowNodeAddress(domain:.model,localID:"group")
        var document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:root,payload:.modelBlock(reference:definition.reference,overrides:[:]))],definitions:[definition]);document.schemaVersion=11
        var next=config;next["layout"] = .string("fixed")
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.setEmbeddingBag(childID:"lookup",configuration:next),to:document,root:root,registry:registry))
        next=config;next["embedding_dim"]=6
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.setEmbeddingBag(childID:"lookup",configuration:next),to:document,root:root,registry:registry))
        next=config;next["padding_idx"]=0
        let result=try WorkflowInlineContainer.apply(.setEmbeddingBag(childID:"lookup",configuration:next),to:document,root:root,registry:registry)
        XCTAssertEqual(try WorkflowInlineContainer.definition(in:result,root:root).body.nodes.first?.config["padding_idx"],0)
        XCTAssertTrue(result.definitions.contains(definition))
    }
}
