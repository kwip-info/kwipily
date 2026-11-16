import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class InlineCanvasRestorationTests: XCTestCase {
    func testLegacyModelAndWorkflowScopesRestoreSavedDraftWithoutConversionOrExecution() throws {
        _ = NSApplication.shared
        for kind in [CanvasTabSession.InlineKind.model, .workflow] {
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: directory) }
            let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            let library = ProjectLibrary(directory: directory.appendingPathComponent("projects"))
            let model = ModelAuthoringDocument(projectID: UUID().uuidString, source: CompositeExamples.parallelClassifier)
            let source: ProjectSourceDocument = kind == .model ? .blockModel(model) : .workflow(try WorkflowCodec.migrateModel(.blockModel(model), projectID: model.projectID).document)
            let record = try library.saveSource(source)
            let registry = ProjectWindowRegistry(presentsWindows: false, makeWorkspace: { WorkspaceStore(rootOverride: repo, storageOverride: directory, detectCapabilities: false) })
            let target = CanvasTabSession.Target.inlineScope(root: .managed(UUID(uuidString: model.projectID)!), kind: kind, path: ["features"])
            let first = try registry.restoreCanvas(target)
            let project = try XCTUnwrap(registry.openControllers.first)
            let workspace = project.session.workspace
            let editor = try XCTUnwrap(kind == .model ? workspace.modelChildWindows?.session.editor : workspace.workflowChildren?.session?.editor)
            editor.moveNodes(ids: ["branch-a"], dx: 9, dy: 0)
            XCTAssertFalse(workspace.isBusy)
            let coordinator = registry.makeRestorationCoordinator()
            XCTAssertTrue(try coordinator.captureAfterPreparation { registry.prepareToTerminate() })
            let saved = try XCTUnwrap(CanvasTabSessionStore(directory: directory.appendingPathComponent("canvas-session")).load())
            XCTAssertTrue(saved.groups.flatMap(\.tabs).contains(target))
            XCTAssertTrue(registry.close(project))
            XCTAssertEqual(registry.projectCount, 0)
            let reopened = try registry.restoreCanvas(target)
            XCTAssertFalse(reopened === first)
            let resumed = try XCTUnwrap(registry.openControllers.first)
            let resumedEditor = try XCTUnwrap(kind == .model ? resumed.session.workspace.modelChildWindows?.session.editor : resumed.session.workspace.workflowChildren?.session?.editor)
            XCTAssertEqual(resumedEditor.path, ["features"])
            XCTAssertEqual(resumedEditor.scope?.nodes.first(where: { $0.id == "branch-a" })?.x, 224)
            XCTAssertFalse(resumed.session.workspace.isBusy)
            XCTAssertTrue(try registry.restoreCanvas(target) === reopened)
            XCTAssertTrue(registry.close(resumed))
            XCTAssertEqual((record.path as NSString).pathExtension, "kwipml")
            XCTAssertNil(try library.fileBackedProjectURL(id: record.id))
        }
    }
}
