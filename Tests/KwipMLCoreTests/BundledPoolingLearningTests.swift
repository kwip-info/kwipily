import XCTest
@testable import KwipMLCore

final class BundledPoolingLearningTests: XCTestCase {
    func testPublishedExamplesResolveWithIndependentGeometry() throws {
        for (id, sample, width) in [("max_pool1d", [2,7], 8), ("max_pool2d", [2,4,5], 12), ("max_pool3d", [2,3,4,5], 24)] {
            let entry = try XCTUnwrap(BundledPoolingComponents.entries[id])
            let learning = try XCTUnwrap(entry.learning)
            XCTAssertGreaterThanOrEqual(learning.paragraphs.count, 7)
            XCTAssertEqual(learning.contentVersion, 2)
            XCTAssertEqual(learning.exampleID, "prepared-" + id.replacingOccurrences(of: "_", with: "") + "-classifier")
            let document = try PreparedWorkflowExample.component(id).document(device: "cpu")
            XCTAssertTrue(document.nodes.contains { node in
                if case .modelPrimitive("reshape", _, let fields) = node.payload { return fields["shape"] == .array(sample.map { .integer(Int64($0)) }) }
                return false
            })
            XCTAssertTrue(document.nodes.contains { node in
                if case .modelPrimitive("linear", _, let fields) = node.payload { return fields["in_features"] == .integer(Int64(width)) }
                return false
            })
        }
    }
    func testUnavailableExampleCannotBeOpenedThroughGenericLauncher() {
        XCTAssertThrowsError(try PreparedWorkflowExample.component("unknown").document())
    }
}
