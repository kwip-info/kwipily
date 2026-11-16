import XCTest
@testable import KwipMLCore

final class BilinearAuthoringConfigurationTests:XCTestCase {
    func testAtomicGeometryRetainsIdentityConnectionsAndPolicies()throws {
        let registry=try ComponentRegistry.builtIn()
        let policy:[String:JSONValue]=["weight":.object(["trainability":.string("frozen")])]
        let node=GraphNode(id:"mix",type:"bilinear",name:"Interactions",x:17,y:29,config:["in1_features":3,"in2_features":5,"out_features":2,"bias":true],parameterSettings:policy)
        var values=node.config;values["in1_features"]=4;values["in2_features"]=6
        let graph=GraphDocument(nodes:[node,.init(id:"out",type:"output")],edges:[.init(source:"mix",target:"out")])
        let result=try GraphCommandEngine.apply(.setBilinear(nodeID:"mix",configuration:values),to:.init(document:graph),registry:registry)
        XCTAssertEqual(result.before.document,graph);XCTAssertEqual(result.after.document.edges,graph.edges)
        let changed=result.after.document.nodes[0]
        XCTAssertEqual(changed.id,node.id);XCTAssertEqual(changed.name,node.name);XCTAssertEqual(changed.x,node.x)
        XCTAssertEqual(changed.parameterSettings,policy);XCTAssertEqual(changed.config,values)
        XCTAssertEqual(try BilinearConfiguration(values:changed.config).parameterCount,50)
    }
    func testBiasPolicyAndLimitsRequireExplicitRepairWithoutMutatingSource()throws {
        let registry=try ComponentRegistry.builtIn()
        let policy:[String:JSONValue]=["bias":.object(["trainability":.string("frozen")])]
        let node=GraphNode(id:"mix",type:"bilinear",config:["in1_features":3,"in2_features":5,"out_features":2,"bias":true],parameterSettings:policy)
        var values=node.config;values["bias"]=false
        XCTAssertThrowsError(try BilinearConfiguration.applying(values,to:node,registry:registry))
        values=node.config;values["in1_features"]=0
        XCTAssertThrowsError(try BilinearConfiguration.applying(values,to:node,registry:registry))
        values=["in1_features":512,"in2_features":512,"out_features":512,"bias":true]
        XCTAssertThrowsError(try BilinearConfiguration.applying(values,to:node,registry:registry))
        var wrong=node;wrong.type="linear"
        XCTAssertThrowsError(try BilinearConfiguration.applying(node.config,to:wrong,registry:registry))
        XCTAssertEqual(node.config["bias"],true)
    }
}
