import Foundation
import XCTest
@testable import KwipMLCore

final class TensorRoutingDeliveryTests: XCTestCase {
    func testCanonicalExampleSerializationPreservesNamedTopology() throws {
        let registry = try ComponentRegistry.builtIn()
        for id in ["split", "chunk", "gather", "einsum"] {
            let graph = try ExampleFlows.tensorRoutingClassifier(componentID: id)
            let bytes = try JSONEncoder().encode(graph)
            let restored = try JSONDecoder().decode(GraphDocument.self, from: bytes)
            XCTAssertEqual(restored, graph)
            XCTAssertTrue(GraphValidator.validate(restored, registry: registry).isValid)
            if let destination = ProcessInfo.processInfo.environment["KWIPML_ROUTING_EXAMPLE_CAPTURE"] {
                let directory = URL(fileURLWithPath: destination, isDirectory: true)
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                try bytes.write(to: directory.appendingPathComponent(id + ".json"), options: .atomic)
            }
        }
    }
    func testRoutingDeclarationsRetainOrderedCeilingsTypedSelectorsAndNoState() throws {
        let registry = try ComponentRegistry.builtIn()
        let declarations: [(String,(ComponentDefinition) throws -> CapabilityDescriptor)] = [
            ("split",TensorRoutingCapabilityBridge.splitDeclaration), ("chunk",TensorRoutingCapabilityBridge.chunkDeclaration),
            ("gather",TensorRoutingCapabilityBridge.gatherDeclaration), ("einsum",TensorRoutingCapabilityBridge.einsumDeclaration)]
        for (id, declaration) in declarations {
            let definition = try XCTUnwrap(registry.component(type:id))
            let descriptor = try declaration(definition)
            XCTAssertEqual(descriptor.ports.map(\.id),definition.inputs+definition.outputs)
            XCTAssertEqual(descriptor.fields,definition.fields)
            XCTAssertTrue(descriptor.lifecycle.state.isEmpty)
            XCTAssertEqual(descriptor.lifecycle.phases,[.initialize,.forward,.backward,.export])
            XCTAssertEqual(descriptor.devices,[.cpu,.mps])
            XCTAssertTrue(CapabilityContractValidator.validate(descriptor).issues.isEmpty)
            if id == "gather" {
                XCTAssertEqual(descriptor.ports.first { $0.id == "indices" }?.schema,
                    .tensor(TensorContract([],dtype:.int64,leadingAxes:.init(minimum:2,maximum:8))))
            }
            var changed = definition
            changed.outputs.reverse()
            if changed.outputs == definition.outputs { changed.outputs.append("unreviewed") }
            XCTAssertThrowsError(try declaration(changed))
        }
    }

    func testDiscoveryRequiresEachExactOptionalDescriptorAndRevokesChangedMetadata() throws {
        XCTAssertTrue(TensorRoutingRuntimeCapabilities().componentIDs.isEmpty)
        for id in ["split","chunk","gather","einsum"] {
            let entry = try XCTUnwrap(BundledTensorRoutingComponents.entries[id])
            let data = try JSONEncoder().encode(entry.runtimeContract)
            XCTAssertEqual(TensorRoutingRuntimeCapabilities(descriptors:[id:data]).componentIDs,[id])
            try TensorRoutingRuntimeCapabilities.verify(data,componentID:id)
            var object = try XCTUnwrap(JSONSerialization.jsonObject(with:data) as? [String:Any])
            object["preservesBatch"] = false
            let changed = try JSONSerialization.data(withJSONObject:object)
            XCTAssertThrowsError(try TensorRoutingRuntimeCapabilities.verify(changed,componentID:id))
            XCTAssertTrue(TensorRoutingRuntimeCapabilities(descriptors:[id:changed]).componentIDs.isEmpty)
            XCTAssertThrowsError(try TensorRoutingRuntimeCapabilities.verify(data,componentID:"unknown"))
        }
    }

    func testExamplesResolveTheirRealNamedPartsAndSelectorTypes() throws {
        let registry = try ComponentRegistry.builtIn()
        for id in ["split","chunk","gather","einsum"] {
            let graph = try ExampleFlows.tensorRoutingClassifier(componentID:id)
            let result = GraphValidator.validate(graph,registry:registry)
            XCTAssertTrue(result.isValid,"\(id): \(result.issues)")
            XCTAssertEqual(graph.nodes.filter { $0.type == id }.count,1)
            XCTAssertEqual(graph.nodes.filter { $0.type == "output" }.count,1)
            if id == "split" {
                XCTAssertEqual(graph.edges.filter { $0.source == "routing" }.map(\.sourcePort),["part2","part0","part1"])
            } else if id == "chunk" {
                XCTAssertEqual(graph.edges.filter { $0.source == "routing" }.map(\.sourcePort),["part4"])
            } else if id == "gather" {
                XCTAssertTrue(graph.edges.contains { $0.source == "positions" && $0.sourcePort == "indices" && $0.target == "routing" && $0.targetPort == "indices" })
            }
        }
        XCTAssertThrowsError(try ExampleFlows.tensorRoutingClassifier(componentID:"unknown"))
    }

    func testPreparedExamplesContainActualDataTrainingAndDistinctRoutingGraphs() throws {
        let factories: [(String,(String,String) throws -> WorkflowDocument)] = [
            ("split",{try WorkflowExamples.preparedSplitClassifier(projectID:$0,device:$1)}),
            ("chunk",{try WorkflowExamples.preparedChunkClassifier(projectID:$0,device:$1)}),
            ("gather",{try WorkflowExamples.preparedGatherClassifier(projectID:$0,device:$1)}),
            ("einsum",{try WorkflowExamples.preparedEinsumClassifier(projectID:$0,device:$1)})]
        for (id,factory) in factories {
            let document = try factory(UUID().uuidString,"cpu")
            XCTAssertTrue(document.nodes.contains { if case .modelPrimitive(let type,_,_) = $0.payload { return type == id }; return false })
            XCTAssertTrue(document.nodes.contains { if case .train = $0.payload { return true }; return false })
            XCTAssertTrue(document.nodes.contains { if case .dataOperator("kwip.data.sourceCSV",_) = $0.payload { return true }; return false })
        }
    }
}
