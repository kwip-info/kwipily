import XCTest
@testable import KwipMLCore

final class SoftmaxConfigurationTests:XCTestCase {
    func testCanonicalAxisAndRankDependentValidation()throws {
        let layout=try BatchedTensorLayout(sample:[2,3])
        for axis:JSONValue in [1,2,-1,-2,1.0] {
            let config=try SoftmaxConfiguration(values:["dim":axis])
            XCTAssertEqual(config.values,["dim":.integer(Int64(axis.integerValue!))])
            XCTAssertEqual(try config.outputLayout(for:layout),layout)
        }
        for axis:JSONValue in [3,-3,-4] {
            let config=try SoftmaxConfiguration(values:["dim":axis])
            XCTAssertThrowsError(try config.outputLayout(for:layout))
        }
    }
    func testMalformedConfigurationRejected() {
        for values:[String:JSONValue] in [[:],["dim":0],["dim":true],["dim":1.5],["dim":.string("1")],
            ["dim":8],["dim":-9],["dim":.integer(Int64.max)],["dim":.number(.infinity)],
            ["dim":.number(.nan)],["dim":1,"extra":1]] {
            XCTAssertThrowsError(try SoftmaxConfiguration(values:values))
        }
    }
    func testGraphPropagationSemanticIdentityAndInvalidAxis()throws {
        let registry=try ComponentRegistry.builtIn()
        let nodes=[GraphNode(id:"input",type:"input",config:["features":2]),GraphNode(id:"linear",type:"linear",config:["in_features":2,"out_features":2]),GraphNode(id:"softmax",type:"softmax",config:["dim":-1.0]),GraphNode(id:"output",type:"output")]
        var graph=GraphDocument(nodes:nodes,edges:zip(nodes,nodes.dropFirst()).map {GraphEdge(source:$0.id,target:$1.id)})
        let result=GraphValidator.validate(graph,registry:registry)
        XCTAssertTrue(result.isValid,"\(result.issues)")
        XCTAssertEqual(result.widths["output"],2)
        XCTAssertEqual(try GraphMigration.effectiveConfiguration(of:graph.nodes[2],schemaVersion:2,registry:registry),["dim":.integer(-1)])
        XCTAssertThrowsError(try GraphMigration.effectiveConfiguration(of:graph.nodes[2],schemaVersion:1,registry:registry))
        let previous=try GraphSemantics.canonicalData(graph,registry:registry)
        graph.nodes[2].config["dim"]=1
        XCTAssertNotEqual(try GraphSemantics.canonicalData(graph,registry:registry),previous)
        for axis:JSONValue in [0,2,-2] {
            graph.nodes[2].config["dim"]=axis
            let invalid=GraphValidator.validate(graph,registry:registry)
            XCTAssertFalse(invalid.isValid)
            XCTAssertNil(invalid.tensorLayouts["output"])
        }
    }

    func testShippedRegistryAndCapabilityDeclareExplicitAxis()throws {
        let registry=try ComponentRegistry.builtIn()
        let component=try XCTUnwrap(registry.component(type:"softmax"))
        XCTAssertEqual(component.inputs,["in"])
        XCTAssertEqual(component.outputs,["out"])
        XCTAssertEqual(component.fields,[ComponentField(key:"dim",title:"Normalization axis",defaultValue:-1,min:-8,max:7)])
        let declaration=try XCTUnwrap(BuiltInCapabilityBridge.declarations(registry:registry).first {$0.id == "softmax"})
        XCTAssertTrue(declaration.lifecycle.state.isEmpty)
        XCTAssertEqual(declaration.adapterID,"kwip.runner.graph-v2.softmax")
        var bad=registry
        let index=try XCTUnwrap(bad.components.firstIndex {$0.id == "softmax"})
        bad.components[index].fields[0].defaultValue=0
        XCTAssertThrowsError(try BuiltInCapabilityBridge.declarations(registry:bad))
    }

}
