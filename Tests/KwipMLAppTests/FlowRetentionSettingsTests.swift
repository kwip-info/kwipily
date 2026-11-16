import XCTest
import KwipMLCore
@testable import KwipMLApp

final class FlowRetentionSettingsTests: XCTestCase {
    @MainActor func testManagedAndExternalLiveOwnersReadSavedPolicyAndAllowDiscardWhileFull() throws {
        for external in [false, true] {
            let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: folder) }
            let managed = ProjectLibrary(directory: folder.appendingPathComponent("managed"))
            let source = try WorkflowExamples.preparedSeparableConv2dClassifier()
            _ = try managed.saveSource(.workflow(source)); _ = try managed.activateFileBackedWorkflow(id: source.projectID)
            let flow = try XCTUnwrap(managed.fileBackedProjectURL(id: source.projectID))
            let library = external ? ProjectLibrary(directory: folder.appendingPathComponent("private"), externalFlowRoot: flow, projectID: source.projectID) : managed
            let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            let owner = try WorkflowAuthoringStore(library: library, projectID: source.projectID, modelRegistry: .builtIn(), dataRegistry: .load(from: repo.appendingPathComponent("runner/dataflow-operators.json")))
            let workspace = WorkspaceStore(rootOverride: repo, storageOverride: folder.appendingPathComponent("app"), detectCapabilities: false)
            workspace.activeProjectID = source.projectID
            let host = try WorkflowWorkspaceSession(workspace: workspace, owner: owner)
            workspace.workflowSession = host
            let settings = workspace.appPreferenceStore
            var preferences = AppPreferences.factory
            preferences.authoringRetention = try .init(maximumFiles: 1, maximumBytes: 1)
            XCTAssertTrue(workspace.setAuthoringRetention(try XCTUnwrap(preferences.authoringRetention)), workspace.appPreferencesError ?? "")
            let original = try Data(contentsOf: flow), root = WorkflowNodeAddress(domain: .model, localID: "convolution")
            XCTAssertFalse(owner.beginDraft(root: root, lease: owner.lease()))
            XCTAssertTrue(owner.error?.contains("Settings") == true, owner.error ?? "")
            XCTAssertEqual(try Data(contentsOf: flow), original)
            XCTAssertTrue(owner.document.drafts.isEmpty)
            preferences.authoringRetention = .factory; XCTAssertTrue(workspace.setAuthoringRetention(try XCTUnwrap(preferences.authoringRetention)), workspace.appPreferencesError ?? "")
            XCTAssertTrue(owner.beginDraft(root: root, lease: owner.lease()), owner.error ?? "")
            let before = owner.document, saved = try Data(contentsOf: flow)
            let directory = flow.deletingLastPathComponent()
            func evidence() throws -> [String: Data] {
                var result: [String: Data] = [:]
                for name in try FileManager.default.contentsOfDirectory(atPath: directory.path) where name.hasPrefix(".kwipflow-draft-") || name.hasPrefix(".kwipflow-publication-") {
                    result[name] = try Data(contentsOf: directory.appendingPathComponent(name))
                }
                return result
            }
            let retained = try evidence()
            preferences.authoringRetention = try .init(maximumFiles: 1, maximumBytes: 1); XCTAssertTrue(workspace.setAuthoringRetention(try XCTUnwrap(preferences.authoringRetention)), workspace.appPreferencesError ?? "")
            var proposed = try XCTUnwrap(before.drafts.first).proposed; proposed.title = "Refused change"
            XCTAssertFalse(owner.updateDraft(proposed, lease: owner.blockLease(root: root)))
            XCTAssertEqual(owner.document, before)
            XCTAssertEqual(try Data(contentsOf: flow), saved)
            XCTAssertEqual(try evidence(), retained)
            XCTAssertTrue(owner.discardDraft(lease: owner.blockLease(root: root)), owner.error ?? "")
            XCTAssertEqual(try evidence(), retained)
            XCTAssertTrue(owner.document.drafts.isEmpty)
            try Data("corrupt preferences".utf8).write(to: settings.fileURL)
            XCTAssertFalse(owner.beginDraft(root: root, lease: owner.lease()))
            XCTAssertTrue(owner.document.drafts.isEmpty)
        }
    }
}
