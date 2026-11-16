import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class StatefulNormalizationCanvasTests: XCTestCase {
    func testLearnedChannelValuesApplyDiscardUndoAndColdDraft() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let repository = TensorLearningDraftRepository(directory: root)
        let session = TensorLearningDraftSession(repository: repository)
        let task = StatefulNormalizationLearningTask.bn1Remembered
        XCTAssertTrue(session.create(projectID: UUID().uuidString, operation: task.rawValue, mode: .guided))
        let graph = GraphDocument(name: "Channel patterns", nodes: [.init(id: "x", type: "input"), .init(id: "op", type: task.componentID, config: try StatefulNormalizationConfiguration(componentID:task.componentID,values:task.configuration).values), .init(id: "out", type: "output")], edges: [.init(id: "a", source: "x", target: "op"), .init(id: "b", source: "op", target: "out")])
        XCTAssertTrue(session.update { $0.graph = graph })
        let canvas = TensorLearningCanvasAdapter(session: session)
        XCTAssertTrue(canvas.initializeNormalizationParameters(nodeID: "op"))
        XCTAssertEqual(session.draft?.parameters?["op"]?["weight"]?.shape, [3])
        XCTAssertTrue(canvas.setGradientTarget("op"))
        XCTAssertTrue(canvas.setPendingParameter(nodeID: "op", name: "bias", index: 1, text: "-0.5"))
        XCTAssertTrue(canvas.applyPendingParameter(nodeID: "op", name: "bias", index: 1))
        XCTAssertEqual(session.draft?.parameters?["op"]?["bias"]?.values[1], -0.5)
        canvas.undo()
        XCTAssertEqual(session.draft?.parameters?["op"]?["bias"]?.values[1], 0)
        canvas.redo()
        XCTAssertEqual(session.draft?.parameters?["op"]?["bias"]?.values[1], -0.5)
        XCTAssertTrue(canvas.setPendingParameter(nodeID: "op", name: "weight", index: 0, text: "2"))
        canvas.discardPendingParameter(nodeID: "op", name: "weight", index: 0)
        XCTAssertNil(canvas.pendingParameter(nodeID: "op", name: "weight", index: 0))
        XCTAssertEqual(try TensorLearningDraft.decode(session.draft!.encoded()), session.draft!)
    }
    func testInactiveFieldsRetainValuesAndExplainTheirEffect() {
        let inactive: [String: JSONValue] = ["affine": false, "track_running_stats": false, "averaging": .string("cumulative")]
        XCTAssertNotNil(ComponentFieldApplicability.inactiveReason(componentID: "batch_norm1d", field: "bias", values: inactive))
        XCTAssertNotNil(ComponentFieldApplicability.inactiveReason(componentID: "batch_norm1d", field: "momentum", values: inactive))
        XCTAssertNotNil(ComponentFieldApplicability.inactiveReason(componentID: "batch_norm1d", field: "averaging", values: inactive))
        XCTAssertNil(ComponentFieldApplicability.inactiveReason(componentID: "batch_norm1d", field: "num_features", values: inactive))
        XCTAssertNil(ComponentFieldApplicability.inactiveReason(componentID: "linear", field: "bias", values: inactive))
    }
}
