import XCTest
@testable import KwipMLCore

final class PoolingBundleMetadataTests: XCTestCase {
    func testEachStatelessWindowBundleOwnsExactControlsAndAssessedLesson() throws {
        let registry = try ComponentRegistry.builtIn()
        let values = [MaxPool1dComponent.metadata, MaxPool2dComponent.metadata, MaxPool3dComponent.metadata, AvgPool1dComponent.metadata]
        for (index, value) in values.enumerated() {
            let average = index == 3
            let dimension = average ? 1 : index + 1
            let id = average ? "avg_pool1d" : "max_pool\(dimension)d"
            let lesson = average ? "assessment.avg-pool1d-windows" : "assessment.max-pool\(dimension)d-windows"
            let entry = try XCTUnwrap(value)
            XCTAssertEqual(entry.definition, registry.component(type: id))
            XCTAssertEqual(entry.definition, BundledPoolingComponents.metadata(id)?.definition)
            XCTAssertEqual(entry.presentation.icon, id + "-v2")
            XCTAssertEqual(entry.presentation.theme, "green")
            XCTAssertEqual(entry.learning?.lessonIDs, [lesson])
            let expected: Set<String> = average ? ["kernel_size", "stride", "padding", "ceil_mode", "count_include_pad"] : ["kernel_size", "stride", "padding", "dilation", "ceil_mode", "return_indices"]
            XCTAssertEqual(Set(entry.definition.fields.map(\.key)), expected)
            let descriptor = try XCTUnwrap(BuiltInCapabilityBridge.declarations(registry: registry).first { $0.id == id })
            XCTAssertEqual(descriptor.learning.lessonIDs, [lesson])
            XCTAssertTrue(descriptor.lifecycle.state.isEmpty)
            if !average { XCTAssertTrue(descriptor.ports.contains { $0.id == "indices" && $0.direction == .output }) }
            var altered = registry
            let position = try XCTUnwrap(altered.components.firstIndex { $0.id == id })
            altered.components[position].title = "Unreviewed"
            XCTAssertThrowsError(try BuiltInCapabilityBridge.declarations(registry: altered))
        }
    }
}
