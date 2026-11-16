import XCTest
import KwipMLCore
@testable import KwipMLApp

final class PanelLayoutActionsTests: XCTestCase {
    @MainActor func testProjectAndArrangementRestorationRejectLateDividerCallbacks() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = WorkspaceStore(rootOverride: directory, detectCapabilities: false)
        store.beginProject(ExampleFlows.parallelClassifier)
        let id = try XCTUnwrap(store.activeProjectID)
        let revision = store.canvasViewRevision
        let sizes: [Double] = [250, 800, 315]
        store.recordPanelSizes(sizes, scope: .modelLibraryInspector, projectID: id, revision: revision)
        XCTAssertEqual(store.panelLayouts[WorkspacePanelScope.modelLibraryInspector.rawValue]?.sizes["library"], 250)
        store.saveWorkspaceArrangement(name: "Wide inspector")
        store.recordPanelSizes([190, 900, 240], scope: .modelLibraryInspector, projectID: id, revision: revision)
        store.applyWorkspaceArrangement(try XCTUnwrap(store.workspaceArrangements.first))
        XCTAssertEqual(store.panelLayouts[WorkspacePanelScope.modelLibraryInspector.rawValue]?.sizes["inspector"], 315)
        store.recordPanelSizes([180, 900, 230], scope: .modelLibraryInspector, projectID: id, revision: revision)
        XCTAssertEqual(store.panelLayouts[WorkspacePanelScope.modelLibraryInspector.rawValue]?.sizes["inspector"], 315, "Pre-apply divider callback must not overwrite restored intent")
        store.persistWorkspaceView()
        store.beginProject(GraphDocument(name: "Other"))
        store.recordPanelSizes(sizes, scope: .modelLibraryInspector, projectID: id, revision: store.canvasViewRevision)
        XCTAssertTrue(store.panelLayouts.isEmpty)
        store.beginProject(try store.projectLibrary.load(id: id), id: id)
        XCTAssertEqual(store.panelLayouts[WorkspacePanelScope.modelLibraryInspector.rawValue]?.sizes["library"], 250)
        let reopened = WorkspaceStore(rootOverride: directory, detectCapabilities: false)
        reopened.beginProject(try reopened.projectLibrary.load(id: id), id: id)
        XCTAssertEqual(reopened.panelLayouts, store.panelLayouts)
    }
    @MainActor func testLegacyNavigationFallbackAndNamedChoiceRestoration() throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let store=WorkspaceStore(rootOverride:directory,detectCapabilities:false)
        store.beginProject(ExampleFlows.parallelClassifier)
        store.legacyCompactNavigationPreference = { false }
        store.installWorkspaceView(WorkspaceViewState())
        XCTAssertFalse(store.compactNavigation, "Missing legacy field reads prior app preference")
        store.saveWorkspaceArrangement(name:"Expanded data",section:"Data")
        store.compactNavigation=true
        store.applyWorkspaceArrangement(try XCTUnwrap(store.workspaceArrangements.first))
        XCTAssertFalse(store.compactNavigation)
        var compact=store.captureWorkspaceView();compact.compactNavigation=true
        store.installWorkspaceView(compact)
        XCTAssertTrue(store.compactNavigation, "An explicit saved choice overrides the legacy fallback")
    }

    @MainActor func testRibbonMigrationAndNewDividerIntentPersistByPaneIdentity() throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let store=WorkspaceStore(rootOverride:directory,detectCapabilities:false)
        XCTAssertTrue(store.beginProject(ExampleFlows.parallelClassifier))
        let id=try XCTUnwrap(store.activeProjectID)
        var old=WorkspaceViewState()
        old.panelLayouts[WorkspacePanelScope.modelLibraryInspector.rawValue] = .init(sizes:["library":211,"canvas":811,"inspector":311])
        old.panelLayouts[WorkspacePanelScope.dataPreparation.rawValue] = .init(sizes:["palette":191,"canvas":711,"inspector":351])
        store.installWorkspaceView(old)
        XCTAssertEqual(store.panelLayouts[WorkspacePanelScope.modelRibbonInspector.rawValue]?.sizes["inspector"],311)
        XCTAssertEqual(store.panelLayouts[WorkspacePanelScope.dataRibbonInspector.rawValue]?.sizes["inspector"],351)
        store.recordPanelSizes([289,900],scope:.modelRibbonInspector,projectID:id,revision:store.canvasViewRevision)
        store.persistWorkspaceView()
        let reopened=WorkspaceStore(rootOverride:directory,detectCapabilities:false)
        XCTAssertTrue(reopened.beginProject(try reopened.projectLibrary.load(id:id),id:id))
        XCTAssertEqual(reopened.panelLayouts[WorkspacePanelScope.modelRibbonInspector.rawValue]?.sizes["inspector"],289)
        XCTAssertEqual(reopened.panelLayouts[WorkspacePanelScope.modelLibraryInspector.rawValue],old.panelLayouts[WorkspacePanelScope.modelLibraryInspector.rawValue])
    }

    @MainActor func testCompositeAndAffineDockChoicesSurviveProjectAndNamedArrangementRestore() throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let store=WorkspaceStore(rootOverride:directory,detectCapabilities:false)
        XCTAssertTrue(store.beginProject(ExampleFlows.parallelClassifier))
        let id=try XCTUnwrap(store.activeProjectID)
        store.recordPanelSizes([305,895],scope:.compositeRibbonInspector,projectID:id,revision:store.canvasViewRevision)
        store.recordPanelSizes([415,785],scope:.affineRibbonDock,projectID:id,revision:store.canvasViewRevision)
        store.saveWorkspaceArrangement(name:"Independent docks")
        let arrangement=try XCTUnwrap(store.workspaceArrangements.first)
        store.recordPanelSizes([400,800],scope:.compositeRibbonInspector,projectID:id,revision:store.canvasViewRevision)
        store.applyWorkspaceArrangement(arrangement)
        XCTAssertEqual(store.panelLayouts[WorkspacePanelScope.compositeRibbonInspector.rawValue]?.sizes["inspector"],305)
        XCTAssertEqual(store.panelLayouts[WorkspacePanelScope.affineRibbonDock.rawValue]?.sizes["dock"],415)
        store.persistWorkspaceView()
        let reopened=WorkspaceStore(rootOverride:directory,detectCapabilities:false)
        XCTAssertTrue(reopened.beginProject(try reopened.projectLibrary.load(id:id),id:id))
        XCTAssertEqual(reopened.panelLayouts,store.panelLayouts)
        XCTAssertNil(reopened.panelLayouts[WorkspacePanelScope.modelRibbonInspector.rawValue])
        XCTAssertNil(reopened.panelLayouts[WorkspacePanelScope.dataRibbonInspector.rawValue])
    }

}
