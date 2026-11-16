import XCTest
@testable import KwipMLCore

final class LinearBundleMetadataTests: XCTestCase {
    func testAuthoredDefinitionPreservesExactFieldsAndReviewedCapabilityShapeStateAndLearningLinks() throws {
        let definition = LinearComponent.definition
        let metadata = try XCTUnwrap(LinearComponent.metadata)
        XCTAssertTrue(metadata.published)
        XCTAssertEqual(metadata.definition, definition)
        XCTAssertEqual(definition.id, "linear")
        XCTAssertEqual(definition.inputs, ["in"]); XCTAssertEqual(definition.outputs, ["out"])
        XCTAssertEqual(definition.fields.map(\.key), ["in_features", "out_features", "bias"])
        XCTAssertEqual(definition.fields.map(\.defaultValue), [2,8,true])
        for field in definition.fields.prefix(2) {
            XCTAssertEqual(field.kind, .integer); XCTAssertEqual(field.min, 1); XCTAssertEqual(field.max, 512)
        }
        XCTAssertEqual(definition.fields.last?.kind, .boolean)
        let descriptor = try BuiltInCapabilityBridge.linearDeclaration(definition)
        XCTAssertEqual(descriptor.version, ContractVersion(1,2,0))
        XCTAssertEqual(descriptor.adapterID, "kwip.runner.graph-v2.linear")
        XCTAssertEqual(descriptor.shapeRuleID, "kwip.shape.linear.v1")
        XCTAssertEqual(descriptor.fields, definition.fields)
        XCTAssertEqual(descriptor.ports.map { $0.schema.tensor?.dimensions }, [[.parameter("in_features")], [.parameter("out_features")]])
        XCTAssertEqual(descriptor.ports.map { $0.schema.tensor?.leadingAxes }, [.init(minimum: 1, maximum: 7), .init(minimum: 1, maximum: 7)])
        XCTAssertEqual(descriptor.lifecycle.state.map(\.kind), [.parameters])
        XCTAssertEqual(descriptor.learning.lessonIDs, ["kwip.course.parallel-classifier.v1", "assessment.affine-composition", "assessment.affine-from-probes", "assessment.affine-backward-chain", "assessment.frozen-standardization"])
        XCTAssertEqual(descriptor.learning.exampleIDs, ["kwip.example.parallel-classifier.v1"])
        for identity in [ModelConstructionLessonIdentity.composition, .recovery(taskSeed: 42), .backwardChain, .frozenStandardization] {
            XCTAssertTrue(descriptor.learning.lessonIDs.contains(identity.assessmentID))
            XCTAssertTrue(try ModelConstructionLessonTasks.publicTask(identity: identity).allowedNodeTypes.contains("linear"))
        }
        let example = try WorkflowExamples.preparedParallelClassifier(device: "cpu")
        XCTAssertTrue(example.nodes.contains { if case .modelPrimitive(let type, _, _) = $0.payload { return type == "linear" }; return false })
    }
    func testExactDeclarationRefusesDefinitionDriftWithoutChangingHistoricalBiasMigration() throws {
        var wrong = LinearComponent.definition
        wrong.fields[0].defaultValue = 3
        XCTAssertThrowsError(try BuiltInCapabilityBridge.linearDeclaration(wrong))
        var graph = GraphDocument(nodes: [.init(id: "i", type: "input", config: ["features": 2]), .init(id: "m", type: "linear", config: ["in_features": 2, "out_features": 2]), .init(id: "o", type: "output")], edges: [.init(source: "i", target: "m"), .init(source: "m", target: "o")])
        graph.schemaVersion = 1
        let linearIndex = try XCTUnwrap(graph.nodes.firstIndex { $0.id == "m" })
        graph.nodes[linearIndex].config.removeValue(forKey: "bias")
        let migrated = try GraphMigration.toCurrent(graph, registry: .builtIn())
        XCTAssertEqual(migrated.nodes.first { $0.id == "m" }?.config["bias"], true)
        XCTAssertEqual(migrated.schemaVersion, 2)
    }
}
