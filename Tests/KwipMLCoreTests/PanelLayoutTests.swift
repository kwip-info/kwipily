import XCTest
@testable import KwipMLCore

final class PanelLayoutTests: XCTestCase {
    func testOldWorkspaceSidecarDecodesWithoutPanelPreferences() throws {
        let old = WorkspaceViewState()
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(old)) as? [String: Any])
        object.removeValue(forKey: "panelLayouts")
        object.removeValue(forKey: "compactNavigation")
        let decoded = try JSONDecoder().decode(WorkspaceViewState.self, from: JSONSerialization.data(withJSONObject: object))
        XCTAssertTrue(decoded.panelLayouts.isEmpty)
        XCTAssertNil(decoded.compactNavigation)
    }
    func testSmallWindowClampsWithoutChangingSavedIntent() throws {
        let scope = WorkspacePanelScope.modelLibraryInspector
        let saved = PanelLayoutPreference(sizes: ["library": 260, "canvas": 700, "inspector": 320])
        let small = WorkspacePanelSizing.sizes(scope: scope, available: 600, preference: saved)
        XCTAssertEqual(small.reduce(0, +), 600, accuracy: 0.001)
        XCTAssertTrue(small.allSatisfy { $0.isFinite && $0 > 0 })
        let large = WorkspacePanelSizing.sizes(scope: scope, available: 1400, preference: saved)
        XCTAssertEqual(large, [260, 820, 320])
        XCTAssertEqual(saved.sizes["library"], 260)
        XCTAssertEqual(saved.sizes["inspector"], 320)
    }
    func testSemanticScopesAndStrictBoundsRoundTrip() throws {
        var state = WorkspaceViewState()
        state.panelLayouts[WorkspacePanelScope.dataPreparation.rawValue] = PanelLayoutPreference(sizes: ["palette": 180, "canvas": 700, "inspector": 310])
        XCTAssertNoThrow(try state.validate())
        XCTAssertEqual(try JSONDecoder().decode(WorkspaceViewState.self, from: JSONEncoder().encode(state)), state)
        state.panelLayouts[WorkspacePanelScope.dataPreparation.rawValue]?.sizes["canvas"] = .nan
        XCTAssertThrowsError(try state.validate())
        state.panelLayouts = ["unknown": PanelLayoutPreference(sizes: ["canvas": 700])]
        XCTAssertThrowsError(try state.validate())
        state.panelLayouts = [WorkspacePanelScope.dataPreparation.rawValue: PanelLayoutPreference(sizes: ["library": 180, "canvas": 700, "inspector": 310])]
        XCTAssertThrowsError(try state.validate(), "A same-count but wrong semantic pane must fail")
    }
    func testDividerHitExcludesCanvasAndWorksWithReversedVerticalCoordinates() {
        XCTAssertTrue(WorkspacePanelSizing.isDividerHit(coordinate: 205.5, paneRanges: [0...205, 206...900]))
        XCTAssertFalse(WorkspacePanelSizing.isDividerHit(coordinate: 400, paneRanges: [0...205, 206...900]))
        XCTAssertTrue(WorkspacePanelSizing.isDividerHit(coordinate: 170.5, paneRanges: [171...700, 0...170]))
        XCTAssertFalse(WorkspacePanelSizing.isDividerHit(coordinate: 500, paneRanges: [171...700, 0...170]))
        XCTAssertFalse(WorkspacePanelSizing.isDividerHit(coordinate: 20, paneRanges: [0...700]))
    }

    func testLearningSectionRetainsExistingCanvasStateWithoutClaimingLessonPanelSupport() throws {
        let state=WorkspaceViewState(activeSection:"Learn",modelCanvas:CanvasViewState(zoom:0.8,scrollX:220,scrollY:30))
        XCTAssertNoThrow(try state.validate())
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let library=WorkspaceArrangementLibrary(directory:directory),id=UUID().uuidString
        try library.saveProjectState(state,projectID:id)
        XCTAssertEqual(try library.projectState(projectID:id),state)
        XCTAssertTrue(state.panelLayouts.isEmpty)
    }

}
