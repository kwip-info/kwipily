import XCTest
@testable import KwipMLCore

final class TensorRoutingExampleOpeningTests: XCTestCase {
    func testEveryCatalogExampleOpensThroughTheSharedDispatcher() throws {
        for id in ["split", "chunk", "gather", "einsum"] {
            let example = PreparedWorkflowExample.component(id)
            let graph = try example.document(device: "mps")
            XCTAssertTrue(graph.nodes.contains { if case .modelPrimitive(let type, _, _) = $0.payload {return type == id}; return false })
            XCTAssertEqual(example.sourceResources.count, 1)
            XCTAssertEqual(example.observationsResource, "workflow-observations")
        }
    }
}
