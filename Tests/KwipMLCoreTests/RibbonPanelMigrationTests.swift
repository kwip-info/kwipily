import XCTest
@testable import KwipMLCore

final class RibbonPanelMigrationTests:XCTestCase {
    func testReorderedPanesUseSemanticInspectorNeverLibraryOrCanvasPosition() throws {
        let legacy:[String:PanelLayoutPreference]=[
            WorkspacePanelScope.modelLibraryInspector.rawValue:.init(sizes:["library":201,"canvas":801,"inspector":319]),
            WorkspacePanelScope.dataPreparation.rawValue:.init(sizes:["palette":181,"canvas":701,"inspector":341])]
        let migrated=WorkspacePanelSizing.ribbonPreferences(legacy)
        XCTAssertEqual(WorkspacePanelScope.modelRibbonInspector.panes.map(\.id),["inspector","canvas"])
        XCTAssertEqual(WorkspacePanelSizing.sizes(scope:.modelRibbonInspector,available:1200,preference:migrated[WorkspacePanelScope.modelRibbonInspector.rawValue]),[319,881])
        XCTAssertEqual(WorkspacePanelSizing.sizes(scope:.dataRibbonInspector,available:1200,preference:migrated[WorkspacePanelScope.dataRibbonInspector.rawValue]),[341,859])
        XCTAssertEqual(migrated[WorkspacePanelScope.modelLibraryInspector.rawValue],legacy[WorkspacePanelScope.modelLibraryInspector.rawValue])
        XCTAssertEqual(WorkspacePanelSizing.ribbonPreferences(migrated),migrated)
        var state=WorkspaceViewState();state.panelLayouts=migrated;XCTAssertNoThrow(try state.validate())
    }
    func testCurrentScopeWinsAndNoInspectorMeansNoInventedFallback() {
        let current=PanelLayoutPreference(sizes:["inspector":275,"canvas":800])
        let values:[String:PanelLayoutPreference]=[
            WorkspacePanelScope.modelRibbonInspector.rawValue:current,
            WorkspacePanelScope.modelInspector.rawValue:.init(sizes:["canvas":800,"inspector":320]),
            WorkspacePanelScope.modelLibrary.rawValue:.init(sizes:["library":220,"canvas":800])]
        XCTAssertEqual(WorkspacePanelSizing.ribbonPreferences(values)[WorkspacePanelScope.modelRibbonInspector.rawValue],current)
        XCTAssertNil(WorkspacePanelSizing.ribbonPreferences([WorkspacePanelScope.modelLibrary.rawValue:values[WorkspacePanelScope.modelLibrary.rawValue]!])[WorkspacePanelScope.modelRibbonInspector.rawValue])
        XCTAssertNil(WorkspacePanelSizing.ribbonPreferences(values)[WorkspacePanelScope.dataRibbonInspector.rawValue])
    }
}
