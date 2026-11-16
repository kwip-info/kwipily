import XCTest
import KwipMLCore
@testable import KwipMLApp

final class GELUCatalogTests: XCTestCase {
    func testMigratedGELUPreservesBothModeCardWithoutDuplicateGenericDisplay() throws {
        XCTAssertEqual(LearnCatalogContent.items.filter { $0.id == "gelu" }.count, 1)
        XCTAssertNotNil(BundledActivationComponents.entries["gelu"]?.learning)
        XCTAssertTrue(LearnCatalogContent.componentExampleEntries.contains { $0.componentID == "gelu" })
        XCTAssertFalse(LearnCatalogContent.displayedComponentExampleEntries.contains { $0.componentID == "gelu" })
        let displayed = LearnCatalogContent.items.map(\.id) + LearnCatalogContent.displayedComponentExampleEntries.map(\.id)
        XCTAssertEqual(displayed.count, Set(displayed).count)
        for mode in [false, true] {
            let graph = try WorkflowExamples.preparedGELUClassifier(device: "cpu", approximateTanh: mode)
            XCTAssertTrue(graph.nodes.contains { if case .modelPrimitive(let type, _, let config) = $0.payload { return type == "gelu" && config["approximate_tanh"] == .bool(mode) }; return false })
        }
    }
}
