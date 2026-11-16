import XCTest
@testable import KwipMLCore

final class CompositePanelLayoutTests:XCTestCase {
    func testCompositeStartsAtThreeHundredAndHasIndependentBoundedIdentity() throws {
        let scope=WorkspacePanelScope.compositeRibbonInspector
        XCTAssertEqual(scope.panes.map(\.id),["inspector","canvas"])
        XCTAssertEqual(scope.panes[0].minimum,240)
        XCTAssertEqual(scope.panes[0].ideal,300)
        XCTAssertEqual(scope.panes[0].maximum,420)
        XCTAssertEqual(WorkspacePanelSizing.sizes(scope:scope,available:1200,preference:scope.initialPreference),[300,900])
        XCTAssertEqual(WorkspacePanelScope.modelRibbonInspector.initialPreference?.sizes["inspector"],280)
        XCTAssertEqual(WorkspacePanelScope.dataRibbonInspector.initialPreference?.sizes["inspector"],290)
        XCTAssertNil(WorkspacePanelScope.modelInspector.initialPreference)
        var state=WorkspaceViewState();state.panelLayouts[scope.rawValue] = .init(sizes:["inspector":335,"canvas":865])
        XCTAssertNoThrow(try state.validate())
        XCTAssertEqual(try JSONDecoder().decode(WorkspaceViewState.self,from:JSONEncoder().encode(state)),state)
        let migrated=WorkspacePanelSizing.ribbonPreferences([WorkspacePanelScope.dataRibbonInspector.rawValue:.init(sizes:["inspector":360,"canvas":840])])
        XCTAssertNil(migrated[scope.rawValue],"Composite must never borrow another workspace's inspector width")
    }
    func testAffineDockHasIndependentInitialWidthAndTabNeutralIdentity() {
        let scope=WorkspacePanelScope.affineRibbonDock
        XCTAssertEqual(scope.panes.map(\.id),["dock","canvas"])
        XCTAssertEqual(WorkspacePanelSizing.sizes(scope:scope,available:1200,preference:scope.initialPreference),[360,840])
        XCTAssertEqual(scope.panes[0].minimum,300)
        XCTAssertEqual(scope.panes[0].maximum,520)
        let chosen=PanelLayoutPreference(sizes:["dock":410,"canvas":790])
        XCTAssertEqual(WorkspacePanelSizing.sizes(scope:scope,available:1200,preference:chosen),[410,790])
    }

}
