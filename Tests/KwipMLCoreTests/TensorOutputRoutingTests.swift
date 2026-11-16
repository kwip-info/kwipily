import XCTest
@testable import KwipMLCore

final class TensorOutputRoutingTests: XCTestCase {
    func testDefaultOutputCompatibilityDoesNotEraseIndicesOrAliasUnknownPorts() throws {
        var result = ValidationResult()
        let values = TypedTensorContract(layout:try .init(sample:[2,3]),dtype:.float32)
        let indices = TypedTensorContract(layout:try .init(sample:[2,3]),dtype:.int64)
        result.outputTensorContracts[.init("pool","indices")] = indices
        result.tensorContracts["pool"] = values
        XCTAssertEqual(result.outputTensorContracts[.init("pool","out")],values)
        XCTAssertEqual(result.outputTensorContracts[.init("pool","indices")],indices)
        XCTAssertNil(result.outputTensorContracts[.init("pool","missing")])
        result.tensorContracts.removeValue(forKey:"pool")
        XCTAssertNil(result.outputTensorContracts[.init("pool","out")])
        XCTAssertEqual(result.outputTensorContracts[.init("pool","indices")],indices)
    }
    func testInputPresentationUsesExactSourcePort() throws {
        var registry = try ComponentRegistry.builtIn()
        let index = try XCTUnwrap(registry.components.firstIndex{$0.id=="relu"})
        registry.components[index].outputs.append("indices")
        let graph = GraphDocument(nodes:[.init(id:"pool",type:"relu"),.init(id:"sink",type:"relu")],
                                  edges:[.init(source:"pool",sourcePort:"indices",target:"sink")])
        var result = ValidationResult()
        result.tensorContracts["pool"] = .init(layout:try .init(sample:[2,3]),dtype:.float32)
        result.outputTensorContracts[.init("pool","indices")] = .init(layout:try .init(sample:[2,3]),dtype:.int64)
        let input = result.tensorPortDescriptor(nodeID:"sink",port:"in",direction:.input,graph:graph,registry:registry)
        XCTAssertTrue(input.declared.contains("int64"))
        let output = result.tensorPortDescriptor(nodeID:"pool",port:"out",direction:.output,graph:graph,registry:registry)
        XCTAssertTrue(output.declared.contains("float32"))
        result.outputTensorContracts.removeValue(forKey:.init("pool","indices"))
        XCTAssertEqual(result.tensorPortDescriptor(nodeID:"sink",port:"in",direction:.input,graph:graph,registry:registry).declared,["Shape unresolved"])
    }
    func testExistingGraphPublishesExplicitDefaultOutputs() throws {
        let result = GraphValidator.validate(ExampleFlows.parallelClassifier,registry:try .builtIn())
        XCTAssertTrue(result.isValid,"\(result.issues)")
        XCTAssertEqual(result.outputTensorContracts.count,result.tensorContracts.count)
        for (node,contract) in result.tensorContracts {
            XCTAssertEqual(result.outputTensorContracts[.init(node,"out")],contract)
        }
    }
    func testValidatorDoesNotSubstituteDefaultForUnresolvedNamedOutput() throws {
        var registry = try ComponentRegistry.builtIn()
        let index = try XCTUnwrap(registry.components.firstIndex{$0.id=="relu"})
        registry.components[index].outputs.append("indices")
        var graph = ExampleFlows.parallelClassifier
        let edge = try XCTUnwrap(graph.edges.firstIndex{$0.id=="e5"})
        graph.edges[edge].sourcePort = "indices"
        let result = GraphValidator.validate(graph,registry:registry)
        XCTAssertFalse(result.isValid)
        XCTAssertNil(result.tensorContracts["merge"])
    }

}
