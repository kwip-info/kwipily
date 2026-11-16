import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

final class WorkflowMetadataTests: XCTestCase {
    @MainActor private func setup(revision: UInt64 = 8) throws -> (URL, ProjectLibrary, WorkflowAuthoringStore) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let library = ProjectLibrary(directory: root)
        let model = ModelAuthoringDocument(projectID: UUID().uuidString, revision: revision, source: CompositeExamples.parallelClassifier)
        var document = try WorkflowCodec.migrateModel(.blockModel(model), projectID: model.projectID).document
        document.nodes.append(.init(address: .init(domain: .data, localID: "features"), name: "Keep source", payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: ["separator": .string(";")])))
        document.nodes.append(.init(address: .init(domain: .control, localID: "setup"), payload: .runSetup(config: ["seed": 123])))
        _ = try library.saveSource(.workflow(document))
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return (root, library, try WorkflowAuthoringStore(library: library, projectID: model.projectID,
            modelRegistry: .builtIn(), dataRegistry: .load(from: repo.appendingPathComponent("runner/dataflow-operators.json"))))
    }

    @MainActor func testRenameCommitsBeforePublicationAndUndoRedoPreserveWholeMixedSource() throws {
        let(root, library, owner) = try setup(); defer { try? FileManager.default.removeItem(at: root) }
        let original = owner.document
        let selection = WorkflowSelection(nodeAddresses: [.init(domain: .data, localID: "features")])
        XCTAssertTrue(owner.select(selection, lease: owner.lease()))
        var published = 0
        let observation = owner.$document.dropFirst().sink { next in
            published += 1
            do {
                let saved = try library.loadSource(id: owner.projectID)
                XCTAssertEqual(saved, .workflow(next), "Disk must contain the new source before document publication")
            } catch { XCTFail("Published source was not saved: \(error)") }
        }
        defer { observation.cancel() }
        XCTAssertTrue(owner.rename("  Mixed research lab  ", lease: owner.lease()))
        var expected = original; expected.name = "  Mixed research lab  "; expected.revision += 1
        XCTAssertEqual(owner.document, expected)
        XCTAssertEqual(owner.selection, selection); XCTAssertTrue(owner.canUndo)
        XCTAssertTrue(owner.undo(lease: owner.lease()))
        var restored = original; restored.revision = expected.revision + 1
        XCTAssertEqual(owner.document, restored); XCTAssertTrue(owner.canRedo)
        XCTAssertTrue(owner.redo(lease: owner.lease()))
        expected.revision = restored.revision + 1
        XCTAssertEqual(owner.document, expected); XCTAssertEqual(owner.selection, selection)
        XCTAssertEqual(published, 3)
    }

    @MainActor func testSameNameIsNoOpAndDoesNotClearRedoOrInvalidateLease() throws {
        let(root, library, owner) = try setup(); defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertTrue(owner.rename("Alternate title", lease: owner.lease()))
        XCTAssertTrue(owner.undo(lease: owner.lease())); XCTAssertTrue(owner.canRedo)
        let before = owner.document, lease = owner.lease(), url = try library.documentURL(id: owner.projectID)
        let bytes = try Data(contentsOf: url)
        var publications = 0
        let observation = owner.$document.dropFirst().sink { _ in publications += 1 }
        defer { observation.cancel() }
        XCTAssertTrue(owner.rename(before.name, lease: lease))
        XCTAssertEqual(owner.document, before); XCTAssertEqual(owner.lease(), lease)
        XCTAssertTrue(owner.canRedo); XCTAssertEqual(publications, 0)
        XCTAssertEqual(try Data(contentsOf: url), bytes)
    }

    @MainActor func testStaleBusyDraftAndCodecBoundRefusalsPreserveSourceAndAuthority() throws {
        let(root, _, owner) = try setup(); defer { try? FileManager.default.removeItem(at: root) }
        let old = owner.lease()
        XCTAssertTrue(owner.rename("Current", lease: old))
        let before = owner.document, current = owner.lease()
        XCTAssertFalse(owner.rename("Stale", lease: old))
        owner.setExecutionBusy(true)
        XCTAssertFalse(owner.rename("Busy", lease: current))
        owner.setExecutionBusy(false)
        XCTAssertFalse(owner.rename(String(repeating: "x", count: 501), lease: current))
        XCTAssertEqual(owner.document, before); XCTAssertEqual(owner.lease(), current)
        XCTAssertTrue(owner.beginDraft(root: .init(domain: .model, localID: "features"), lease: current))
        let draft = owner.document, draftLease = owner.lease()
        XCTAssertFalse(owner.rename("Would stale the draft base", lease: draftLease))
        XCTAssertEqual(owner.document, draft); XCTAssertEqual(owner.lease(), draftLease)
        owner.invalidate()
        XCTAssertFalse(owner.rename("Retired", lease: owner.lease()))
        XCTAssertEqual(owner.document, draft)
    }

    @MainActor func testSaveFailurePreservesLeasePublicationAndBothHistoryBranches() throws {
        let(root, library, owner) = try setup(); defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertTrue(owner.rename("One", lease: owner.lease()))
        XCTAssertTrue(owner.rename("Two", lease: owner.lease()))
        XCTAssertTrue(owner.undo(lease: owner.lease()))
        let before = owner.document, lease = owner.lease()
        XCTAssertTrue(owner.canUndo); XCTAssertTrue(owner.canRedo)
        let url = try library.documentURL(id: owner.projectID), backup = root.appendingPathComponent("backup.json")
        try FileManager.default.moveItem(at: url, to: backup)
        try FileManager.default.createSymbolicLink(at: url, withDestinationURL: backup)
        var publications = 0
        let observation = owner.$document.dropFirst().sink { _ in publications += 1 }
        defer { observation.cancel() }
        XCTAssertFalse(owner.rename("Never committed", lease: lease))
        XCTAssertEqual(owner.document, before); XCTAssertEqual(owner.lease(), lease)
        XCTAssertTrue(owner.canUndo); XCTAssertTrue(owner.canRedo); XCTAssertEqual(publications, 0)
        XCTAssertNotNil(owner.error)
        try FileManager.default.removeItem(at: url); try FileManager.default.moveItem(at: backup, to: url)
        XCTAssertTrue(owner.redo(lease: lease)); XCTAssertEqual(owner.document.name, "Two")
    }
}
