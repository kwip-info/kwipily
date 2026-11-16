import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ModuleCompositionLearningAppTests: XCTestCase {
    private var repository: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    func testNestedCanvasSessionStartsWithAnActualRootAdapter() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let drafts = TensorLearningDraftSession(repository: .init(directory: root))
        XCTAssertTrue(drafts.create(projectID: UUID().uuidString, operation: ModuleCompositionLearningTask.orderName.rawValue, mode: .guided))
        let session = try ModuleCompositionLearningCanvasSession(drafts: drafts, registry: .builtIn(), isCurrent: { true }, isBusy: { false })
        XCTAssertNotNil(session.activeCanvas)
        XCTAssertEqual(session.activePath, [])
        XCTAssertEqual(session.paths, [[]])
        session.invalidate()
        XCTAssertNil(session.activeCanvas)
    }
    func testTabsPreserveUnappliedChildAndColdRestoreItsExactScope()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let drafts=TensorLearningDraftSession(repository:.init(directory:root))
        let project=UUID().uuidString
        XCTAssertTrue(drafts.create(projectID:project,operation:ModuleCompositionLearningTask.buildReuse.rawValue,mode:.guided))
        let session=try ModuleCompositionLearningCanvasSession(drafts:drafts,registry:.builtIn(),isCurrent:{true},isBusy:{false})
        let outer=try XCTUnwrap(drafts.draft?.compositionModel?.source.root.instances.first)
        session.open([outer.id]);XCTAssertEqual(session.activePath,[outer.id])
        XCTAssertTrue(session.apply(),session.statusMessage)
        let model=try XCTUnwrap(drafts.draft?.compositionModel)
        let applied=try XCTUnwrap(model.source.root.instances.first)
        let body=try XCTUnwrap(model.source.definitions.first {$0.reference == applied.definition})
        let child=try XCTUnwrap(body.body.instances.first)
        let path=[outer.id,child.id]
        session.open(path);XCTAssertEqual(session.activePath,path)
        let pending=try XCTUnwrap(drafts.draft?.compositionModel?.drafts.first)
        session.open([outer.id])
        XCTAssertEqual(session.activePath,path,"Changing tabs cannot silently abandon the child draft")
        session.close(path);XCTAssertTrue(session.paths.contains(path))
        let reopened=TensorLearningDraftSession(repository:.init(directory:root))
        XCTAssertTrue(reopened.open(projectID:project))
        let restored=try ModuleCompositionLearningCanvasSession(drafts:reopened,registry:.builtIn(),isCurrent:{true},isBusy:{false})
        XCTAssertEqual(restored.activePath,path)
        XCTAssertEqual(reopened.draft?.compositionModel?.drafts.first,pending)
        XCTAssertTrue(session.discard());session.close(path);XCTAssertEqual(session.activePath,[])
        XCTAssertFalse(session.paths.contains(path))
    }
    func testProtectedNestedLessonUsesSingleDraftSourceAndColdRestore() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        let lesson = TensorLearningStore(workspace: workspace)
        for task in ModuleCompositionLearningTask.allCases {
            XCTAssertTrue(lesson.start(operation: task.rawValue), lesson.statusMessage)
            XCTAssertNotNil(lesson.compositionCanvas, "Nested lesson needs its actual source canvas")
            XCTAssertNil(lesson.canvas, "Ordinary forward graph must not become a second editable authority")
            XCTAssertTrue(lesson.draft?.graph.nodes.isEmpty == true)
            XCTAssertFalse(lesson.canSubmit, "Incomplete source and missing owned discovery cannot be graded")
            XCTAssertNotNil(lesson.runtimeReadinessError)
            XCTAssertTrue(lesson.save(), lesson.statusMessage)
            let id = try XCTUnwrap(workspace.activeProjectID)
            let cold = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
            let reopened = TensorLearningStore(workspace: cold)
            XCTAssertTrue(cold.beginProject(workspace.graph, id: id), cold.statusMessage)
            XCTAssertEqual(reopened.draft, lesson.draft)
            XCTAssertNotNil(reopened.compositionCanvas)
            XCTAssertNil(reopened.canvas)
            XCTAssertFalse(reopened.canSubmit)
        }
    }
    func testWorkedExamplesResolveToLessonButDoNotGrantExecution() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        let lesson = TensorLearningStore(workspace: workspace)
        for task in ModuleCompositionLearningTask.allCases {
            XCTAssertTrue(lesson.start(operation: "prepared-" + task.rawValue), lesson.statusMessage)
            XCTAssertEqual(lesson.draft?.operation, task.rawValue)
            XCTAssertNotNil(lesson.compositionCanvas)
            XCTAssertNotNil(lesson.runtimeReadinessError)
            XCTAssertFalse(lesson.canSubmit)
        }
    }
}
