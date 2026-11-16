import XCTest
@testable import KwipMLCore

final class ModelTensorProjectionTests:XCTestCase {
    func testExposedBlockPortsResolveThroughNestedBoundaries()throws {
        var source=CompositeExamples.parallelClassifier
        let inner=source.definitions[0]
        let wrapper=CompositeDefinition(reference:.init("wrapper"),title:"Wrapper",body:.init(instances:[.init(id:"inner",definition:inner.reference)]),inputs:[.init("in",targets:[.init("inner","in")])],outputs:[.init("out",source:.init("inner","out"))])
        source.definitions.append(wrapper)
        source.root.instances[0].definition=wrapper.reference
        let registry=try ComponentRegistry.builtIn()
        let snapshot=try ModelExecutionSnapshotBuilder.build(.blockModel(.init(projectID:UUID().uuidString,revision:1,source:source)),registry:registry)
        let projection=ModelTensorProjection(snapshot:snapshot,registry:registry)
        for (path,node) in [([String](),"features"),(["features"],"inner")] {
            XCTAssertEqual(projection.port(instancePath:path,nodeID:node,port:"in",direction:.input).declared,["[N, 2]","float32"])
            XCTAssertEqual(projection.port(instancePath:path,nodeID:node,port:"out",direction:.output).declared,["[N, 16]","float32"])
        }
    }
    func testExpandedOriginsResolveInputOutputWithoutMeasuredClaims()throws {
        let registry=try ComponentRegistry.builtIn()
        let snapshot=try ModelExecutionSnapshotBuilder.build(.blockModel(.init(projectID:UUID().uuidString,revision:1,source:CompositeExamples.parallelClassifier)),registry:registry)
        let projection=ModelTensorProjection(snapshot:snapshot,registry:registry)
        var nestedLinears=0
        for (id,origin) in snapshot.origins {
            guard let node=snapshot.graph.nodes.first(where:{$0.id == id}),node.type == "linear" else {continue}
            if !origin.instancePath.isEmpty {nestedLinears+=1}
            let input=projection.port(instancePath:origin.instancePath,nodeID:origin.localNodeID,port:"in",direction:.input)
            let output=projection.port(instancePath:origin.instancePath,nodeID:origin.localNodeID,port:"out",direction:.output)
            XCTAssertEqual(input.declared,["[N, \(node.config["in_features"]!.integerValue!)]","float32"])
            XCTAssertEqual(output.declared,["[N, \(node.config["out_features"]!.integerValue!)]","float32"])
            XCTAssertEqual(projection.inputLayout(instancePath:origin.instancePath,nodeID:origin.localNodeID,port:"in")?.sample.dimensions,[Int(node.config["in_features"]!.integerValue!)])
            XCTAssertEqual(projection.inputContract(instancePath:origin.instancePath,nodeID:origin.localNodeID,port:"in")?.dtype,.float32)
            XCTAssertTrue(input.measured.isEmpty)
            XCTAssertTrue(output.measured.isEmpty)
        }
        XCTAssertGreaterThan(nestedLinears,0)
        XCTAssertEqual(projection.port(instancePath:["missing"],nodeID:"branch-a",port:"out",direction:.output).kind,.unknown)
        XCTAssertEqual(projection.port(instancePath:[],nodeID:"input",port:"wrong",direction:.output).kind,.unknown)
    }
}
