import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class BagGradientArchiveTests: XCTestCase {
    func testCapturedBagInputTargetsRetainTheirActualRoleAndVersionedRecipe() throws {
        let fixtures = BagWorkflowArchiveTests()
        for port in ["indices", "offsets", "per_sample_weights"] {
            var doc = try fixtures.fixture(packed: true, weighted: true)
            let old = try fixtures.compile(doc)
            let source = try XCTUnwrap(old.model.graph.nodes.first { $0.type == "bag_input" }).id
            let table = try XCTUnwrap(old.model.graph.nodes.first { $0.type == "embedding_bag" }).id
            let request: JSONValue = .object(["id": .string("influence"), "output": .object(["node": .string(table), "port": .string("out")]), "partition": .string("validation"), "row_start": 0, "row_count": 1, "reduction": .string("sum"), "upstream": .null, "targets": .array([.object(["kind": .string("input"), "node": .string(source), "port": .string(port)])]), "unavailable": .string("reportUnavailable")])
            let configuration = try GradientPolicyConfiguration(values: ["requests": .array([request])])
            let address = WorkflowNodeAddress(domain: .control, localID: "influence")
            doc.nodes.append(.init(address: address, payload: .gradientPolicyRecipe(type: "gradient_policy", configurationVersion: 1, config: configuration.values)))
            doc.edges.append(.init(id: "influence-train", from: .init(address: address, port: "out"), to: .init(address: WorkflowExamples.trainAddress, port: "gradientPolicy"), kind: .gradientPolicyRecipe))
            let plan = try fixtures.compile(doc)
            let archived = try WorkflowArchivePlan(plan, compilerContractVersion: 11)
            XCTAssertEqual(archived.gradientPolicyRecipe?.configuration, configuration.recipeValue)
            XCTAssertEqual(try JSONDecoder().decode(WorkflowArchivePlan.self, from: WorkflowArchiveCodec.encode(archived)), archived)
        }
    }
}
