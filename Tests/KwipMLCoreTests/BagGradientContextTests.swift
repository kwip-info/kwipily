import Foundation
import XCTest
@testable import KwipMLCore

final class BagGradientContextTests: XCTestCase {
    private func snapshot(packed: Bool, weighted: Bool) throws -> (ModelExecutionSnapshot, ComponentRegistry) {
        let fixtures = BagCompositionSourceTests(), registry = try fixtures.registry()
        let source = fixtures.project(packed: packed, weighted: weighted)
        let snapshot = try ModelExecutionSnapshotBuilder.build(.blockModel(.init(projectID: UUID().uuidString, source: source, schemaVersion: 2)), registry: registry, context: .preparedWorkflow)
        return (snapshot, registry)
    }
    private func policy(model: ModelExecutionSnapshot, node: String, port: String, output: String? = nil) throws -> GradientPolicyConfiguration {
        let table = try XCTUnwrap(model.origins.first { $0.value.localNodeID == "table" }).key
        let request: JSONValue = .object(["id": .string("bag-influence"), "output": .object(["node": .string(output ?? table), "port": .string("out")]), "partition": .string("validation"), "row_start": 0, "row_count": 2, "reduction": .string("sum"), "upstream": .null, "targets": .array([.object(["kind": .string("input"), "node": .string(node), "port": .string(port)])]), "unavailable": .string("reportUnavailable")])
        return try .init(values: ["requests": .array([request])])
    }
    func testEveryEnabledBagRoleIsAQualifiedInputTargetWithoutInventingTensorGeometry() throws {
        for packed in [false, true] { for weighted in [false, true] {
            let (model, registry) = try snapshot(packed: packed, weighted: weighted)
            let source = try XCTUnwrap(model.graph.nodes.first { $0.type == "bag_input" })
            let configuration = try BagSourceConfiguration(values: source.config)
            for port in configuration.outputPorts {
                let selected = try policy(model: model, node: source.id, port: port)
                try GradientWorkflowRecipeContext.validate(policy: selected, trainability: nil, model: model, registry: registry, context: .preparedWorkflow)
                let validation = GraphValidator.validate(model.graph, registry: registry, context: .preparedWorkflow)
                XCTAssertNotNil(validation.outputBagContracts[.init(source.id, port)])
                XCTAssertNil(validation.outputTensorContracts[.init(source.id, port)])
            }
        } }
    }
    func testDisabledBagPortsForeignOwnersAndIntermediateInputsRemainRefused() throws {
        let (model, registry) = try snapshot(packed: false, weighted: false)
        let source = try XCTUnwrap(model.graph.nodes.first { $0.type == "bag_input" })
        let table = try XCTUnwrap(model.graph.nodes.first { $0.type == "embedding_bag" })
        for (node, port) in [(source.id, "offsets"), (source.id, "per_sample_weights"), (source.id, "out"), ("foreign", "indices"), (table.id, "out")] {
            XCTAssertThrowsError(try GradientWorkflowRecipeContext.validate(policy: policy(model: model, node: node, port: port), trainability: nil, model: model, registry: registry, context: .preparedWorkflow))
        }
    }
}
