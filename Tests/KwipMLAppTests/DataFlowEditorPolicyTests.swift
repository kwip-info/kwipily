import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class DataFlowEditorPolicyTests: XCTestCase {
    private func fixture(_ workspace:WorkspaceStore)->DataFlowProject {
        DataFlowProject(name:"Unsolved lesson",graph:DataFlowDocument(nodes:[DataFlowNode(id:"source",operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")])]),editor:["source":DataFlowEditorNode(name:"Fixture observations",x:40,y:100)])
    }
    func testLessonDoesNotLoadExistingPreparationOrCacheAndHasOnlyAllowedOperators() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent(),storageOverride:root,detectCapabilities:false);workspace.activeProjectID=UUID().uuidString
        let ordinary=DataFlowWorkspaceStore(workspace:workspace);ordinary.save();ordinary.setCachePolicy(.reuse)
        let provided=fixture(workspace)
        let lesson=DataFlowWorkspaceStore(workspace:workspace,lessonProject:provided)
        XCTAssertEqual(lesson.project,provided);XCTAssertEqual(lesson.policy,.normalizationLesson)
        XCTAssertEqual(Set(lesson.registry.components.map(\.id)),NormalizationLessonTask.allowedOperators.intersection(DataFlowWorkspaceStore.editableTypes))
        XCTAssertEqual(lesson.cachePolicy,.off);XCTAssertNil(lesson.lastCacheReference);XCTAssertNil(lesson.manifestURL)
        XCTAssertFalse(lesson.isInspectingSource);XCTAssertFalse(lesson.isRestoringResults)
        XCTAssertTrue(lesson.validation.isValid)
        XCTAssertEqual(lesson.availableColumnSchema(for:"source").map(\.name),["x","y"])
        XCTAssertEqual(lesson.availableColumnSchema(for:"source").map(\.dtype),["string","string"])
    }
    func testDirectPreparationEntryPointsCannotEscapeLessonPolicy() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent(),storageOverride:root,detectCapabilities:false);workspace.activeProjectID=UUID().uuidString
        let lesson=DataFlowWorkspaceStore(workspace:workspace,lessonProject:fixture(workspace));lesson.selectNode("source")
        let original=lesson.project;var launches=0
        lesson.sourceInspectionExecute={_,_,_,_,_,_,_,_ in launches+=1;return .init(cancel:{})}
        lesson.resultRestorationExecute={_,_,_,_,_,_,_ in launches+=1;return .init(cancel:{})}
        lesson.bindSource("/private/unrelated.csv",nodeID:"source");lesson.chooseSource();lesson.inspectSource(id:"source")
        lesson.runPreparation();lesson.exportPortableFlow();lesson.save();lesson.reopen();lesson.loadExample(columns:["x","y"])
        lesson.loadBundledFittedExample();lesson.loadBundledSensorExample();lesson.restoreCachePreferences();lesson.setCachePolicy(.reuse);lesson.persistCachePreferences();lesson.restoreCompletedOutputs()
        XCTAssertThrowsError(try lesson.preparationCacheArguments())
        XCTAssertThrowsError(try lesson.readCompletedReference())
        XCTAssertThrowsError(try lesson.persistCompletedResult(manifest:root.appendingPathComponent("absent"),graph:original.graph,bindings:[:],digests:[:]))
        XCTAssertThrowsError(try lesson.acceptCacheReport(manifestURL:root.appendingPathComponent("absent"),report:nil))
        XCTAssertEqual(lesson.project,original);XCTAssertEqual(launches,0);XCTAssertFalse(lesson.resultsCurrent)
        XCTAssertFalse(FileManager.default.fileExists(atPath:lesson.saveURL.path));XCTAssertFalse(lesson.history.canUndo)
        XCTAssertNil(lesson.preparationWorker);XCTAssertFalse(workspace.isPreparingData)
    }
    func testAllowedEditingAndUndoWorkButSourceConfigAssessmentAndStaleProjectAreGuarded() {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent(),storageOverride:root,detectCapabilities:false);workspace.activeProjectID=UUID().uuidString
        let lesson=DataFlowWorkspaceStore(workspace:workspace,lessonProject:fixture(workspace))
        lesson.addNode(type:"kwip.data.filter",x:100,y:100);XCTAssertEqual(lesson.project.graph.nodes.count,1)
        lesson.selectNode("source");lesson.configure(["path":.string("/private/data")],nodeID:"source")
        XCTAssertEqual(lesson.project.graph.nodes.first?.config,["separator":.string(",")])
        for retired in RetiredDataOperatorMigration.retired {lesson.addNode(type:retired,x:230,y:100)}
        XCTAssertEqual(lesson.project.graph.nodes.count,1,"Historical lesson schemas never re-enable retired authoring.")
        lesson.selectNode("source");lesson.rename("Reviewed fixture observations")
        let cast="source"
        XCTAssertEqual(lesson.availableColumnSchema(for:cast).map(\.name),["x","y"])

        let connected=lesson.project;lesson.undo();XCTAssertEqual(lesson.project.editor["source"]?.name,"Fixture observations");lesson.redo();XCTAssertEqual(lesson.project,connected)
        workspace.setLearningAssessmentActive(true);lesson.moveNodes(ids:[cast],dx:10,dy:10);XCTAssertEqual(lesson.project,connected)
        workspace.setLearningAssessmentActive(false);workspace.activeProjectID=UUID().uuidString
        lesson.moveNodes(ids:[cast],dx:10,dy:10);XCTAssertEqual(lesson.project,connected)
    }
}
