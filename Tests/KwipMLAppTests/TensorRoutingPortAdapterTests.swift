import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TensorRoutingPortAdapterTests: XCTestCase {
    @MainActor func testUnavailablePartCannotFallBackToDeclaredCeiling() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let store = WorkspaceStore(rootOverride: root, detectCapabilities: false)
        store.graph = try ExampleFlows.tensorRoutingClassifier(componentID: "chunk")
        XCTAssertEqual(store.canvasPortDescriptor(nodeID: "routing", port: "part5", direction: .output).kind, .unknown)
        XCTAssertEqual(store.canvasPortDescriptor(nodeID: "routing", port: "part4", direction: .output).declared, ["[N, 2, 1]", "float32"])
    }
}
