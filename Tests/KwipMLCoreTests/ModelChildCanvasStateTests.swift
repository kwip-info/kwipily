import XCTest
@testable import KwipMLCore

final class ModelChildCanvasStateTests: XCTestCase {
    func testCanonicalInstancePathsAreDistinctAndBounded() throws {
        let first = try ModelChildCanvasKey.make(path: ["a/b", "c"])
        let second = try ModelChildCanvasKey.make(path: ["a", "b/c"])
        XCTAssertNotEqual(first, second)
        XCTAssertTrue(ModelChildCanvasKey.isValid(first))
        XCTAssertFalse(ModelChildCanvasKey.isValid("block:not base64"))
        XCTAssertThrowsError(try ModelChildCanvasKey.make(path: []))
        XCTAssertThrowsError(try ModelChildCanvasKey.make(path: Array(repeating: "a", count: 9)))
    }
    func testLegacyDecodeNamedAndProjectRoundTripAndBounds() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let library = WorkspaceArrangementLibrary(directory: root)
        let key = try ModelChildCanvasKey.make(path: ["features"])
        var state = WorkspaceViewState()
        state.blockCanvasStates[key] = .init(zoom: 0.4, scrollX: 91, scrollY: 82, presentation: .init(flowDirection: .topToBottom, nodeDetail: .small))
        let saved = try library.save(name: "Child view", state: state)
        try library.saveProjectState(state, projectID: saved.id)
        XCTAssertEqual(try library.list().first?.state, state)
        XCTAssertEqual(try library.projectState(projectID: saved.id), state)
        var json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(state)) as! [String: Any]
        json.removeValue(forKey: "blockCanvasStates")
        let legacy = try JSONDecoder().decode(WorkspaceViewState.self, from: JSONSerialization.data(withJSONObject: json))
        XCTAssertTrue(legacy.blockCanvasStates.isEmpty)
        state.blockCanvasStates[key]?.zoom = 0.01
        XCTAssertNoThrow(try state.validate())
        state.blockCanvasStates[key]?.zoom = 0.005
        XCTAssertThrowsError(try state.validate())
        state.blockCanvasStates = ["not a path": .init()]
        XCTAssertThrowsError(try state.validate())
    }
}
