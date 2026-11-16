import XCTest
@testable import KwipMLCore

final class MatrixBundleMetadataTests: XCTestCase {
    func testExactAuthoredParameterlessDefinitionAndRealLearningLinks() throws {
        let entry = try XCTUnwrap(MatrixMultiplyComponent.metadata)
        let definition = MatrixMultiplyComponent.definition
        XCTAssertEqual(entry.definition, definition)
        let additiveIDs:Set<String>=["bag_input","embedding_bag","dataset_source","data_loader"]
        XCTAssertEqual(fixtureModelComponentsBeforeTextExtensions(try .builtIn()).filter { $0.id != "initialization" && $0.id != "gradient_policy" && $0.id != "trainability" && !additiveIDs.contains($0.id) }.count, 83)
        let currentComponents=fixtureModelComponentsBeforeTextExtensions(try .builtIn())
        XCTAssertEqual(Set(currentComponents.filter {additiveIDs.contains($0.id)}.map(\.id)),additiveIDs)
        XCTAssertEqual(currentComponents.filter { !["initialization","gradient_policy","trainability"].contains($0.id) }.count,87)
        XCTAssertEqual(definition.inputs, ["a", "b"])
        XCTAssertEqual(definition.outputs, ["out"])
        XCTAssertTrue(definition.fields.isEmpty)
        XCTAssertEqual(entry.presentation.icon, "matmul-v2")
        XCTAssertEqual(entry.learning?.exampleID, "prepared-matrix-classifier")
        let descriptor = try BuiltInCapabilityBridge.matrixDeclaration(definition)
        XCTAssertEqual(descriptor.adapterID, "kwip.runner.graph-v2.matrixMultiply")
        XCTAssertEqual(descriptor.shapeRuleID, "kwip.shape.matmul.v1")
        XCTAssertEqual(descriptor.learning.lessonIDs, ["assessment.matrix-contraction", "assessment.matrix-broadcast-gradients"])
        XCTAssertTrue(descriptor.lifecycle.state.isEmpty)
        for field in ["id", "title", "category"] {
            var wrong = definition
            if field == "id" { wrong.id = "wrong" }
            if field == "title" { wrong.title = "wrong" }
            if field == "category" { wrong.category = "wrong" }
            XCTAssertThrowsError(try BuiltInCapabilityBridge.matrixDeclaration(wrong), field)
        }
    }
}
