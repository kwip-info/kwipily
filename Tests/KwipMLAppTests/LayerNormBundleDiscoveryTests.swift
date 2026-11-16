import XCTest
import KwipMLCore
@testable import KwipMLApp
final class LayerNormBundleDiscoveryTests: XCTestCase {
    func testAuthoredLearningIsDiscoverableWithoutDuplicatingTheTwoScopeCard() throws {
        XCTAssertEqual(LearnCatalogContent.componentExampleEntries.filter { $0.componentID == "layer_norm" }.count, 1)
        XCTAssertFalse(LearnCatalogContent.displayedComponentExampleEntries.contains { $0.componentID == "layer_norm" })
        let flow = try PreparedWorkflowExample.component("layer_norm").document(device: "cpu")
        XCTAssertTrue(flow.nodes.contains { if case .modelPrimitive(let type, _, let config) = $0.payload { return type == "layer_norm" && config["normalized_shape"] == .array([3]) }; return false })
    }
}
