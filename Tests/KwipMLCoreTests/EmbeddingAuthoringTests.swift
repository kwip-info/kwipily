import XCTest
@testable import KwipMLCore

final class EmbeddingAuthoringTests:XCTestCase {
    func testAtomicGraphEditPreservesIdentityWeightsAndRejectsConflict()throws {
        let registry=try ComponentRegistry.builtIn()
        let config=Dictionary(uniqueKeysWithValues:try XCTUnwrap(registry.component(type:"embedding")).fields.map {($0.key,$0.defaultValue)})
        let settings:[String:JSONValue]=["weight":.object(["trainability":.string("frozen")])]
        let node=GraphNode(id:"lookup",type:"embedding",name:"Categories",config:config,parameterSettings:settings)
        let graph=GraphDocument(schemaVersion:2,nodes:[node])
        var next=config;next["num_embeddings"]=32;next["embedding_dim"]=6;next["padding_idx"] = -1
        let result=try GraphCommandEngine.apply(.setEmbedding(nodeID:node.id,configuration:next),to:.init(document:graph),registry:registry)
        let changed=try XCTUnwrap(result.after.document.nodes.first)
        XCTAssertEqual(changed.id,node.id);XCTAssertEqual(changed.name,node.name);XCTAssertEqual(changed.parameterSettings,settings)
        XCTAssertEqual(changed.config["padding_idx"],31)
        next["sparse"]=true;next["scale_grad_by_freq"]=true
        XCTAssertThrowsError(try GraphCommandEngine.apply(.setEmbedding(nodeID:node.id,configuration:next),to:.init(document:graph),registry:registry))
        XCTAssertEqual(graph.nodes,[node])
    }
    func testInlineExposedDimensionIsProtectedWhileOtherFieldsCanChange()throws {
        let registry=try ComponentRegistry.builtIn()
        let config=Dictionary(uniqueKeysWithValues:try XCTUnwrap(registry.component(type:"embedding")).fields.map {($0.key,$0.defaultValue)})
        let field=try XCTUnwrap(registry.component(type:"embedding")?.fields.first {$0.key=="embedding_dim"})
        let definition=CompositeDefinition(reference:.init("lookup-container"),title:"Lookup",body:.init(nodes:[.init(id:"lookup",type:"embedding",config:config)],configurationVersion:2),inputs:[.init("in",targets:[.init("lookup","in")])],outputs:[.init("out",source:.init("lookup","out"))],parameters:[.init(field,targets:[.init("lookup","embedding_dim")])],composition:.init(kind:.graph,children:[.init(localID:"lookup")]))
        let root=WorkflowNodeAddress(domain:.model,localID:"container")
        var source=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:root,payload:.modelBlock(reference:definition.reference,overrides:[:]))],definitions:[definition])
        source.schemaVersion=11
        var changed=config;changed["embedding_dim"]=12
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.setEmbedding(childID:"lookup",configuration:changed),to:source,root:root,registry:registry))
        changed=config;changed["padding_idx"]=0
        let result=try WorkflowInlineContainer.apply(.setEmbedding(childID:"lookup",configuration:changed),to:source,root:root,registry:registry)
        let updated=try WorkflowInlineContainer.definition(in:result,root:root)
        XCTAssertEqual(updated.body.nodes.first?.config["padding_idx"],0)
        XCTAssertTrue(result.definitions.contains(definition))
        XCTAssertEqual(source.definitions,[definition])
    }
}
