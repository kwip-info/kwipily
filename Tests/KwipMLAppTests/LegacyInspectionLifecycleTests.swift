import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LegacyInspectionLifecycleTests: XCTestCase {
    @MainActor func testSuccessfulCSVProjectLoadRoutesToRequestedSectionWhileInspectionStarts() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let csv = root.appendingPathComponent("source.csv"); try "a,b\n1,2\n".write(to: csv, atomically: true, encoding: .utf8)
        let store = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        var graph = ExampleFlows.parallelClassifier; graph.execution = ExecutionSettings(datasetCSV: csv.path)
        let record = try store.projectLibrary.save(graph)
        store.openManagedProject(record, section: "Data")
        XCTAssertEqual(store.activeProjectID, record.id)
        XCTAssertTrue(store.isInspectingData)
        XCTAssertEqual(store.activeSection, "Data")
        let deadline = Date().addingTimeInterval(10)
        while store.isInspectingData && Date() < deadline { try await Task.sleep(nanoseconds: 25_000_000) }
        XCTAssertFalse(store.isInspectingData)
        let section = store.activeSection, project = store.activeProjectID
        store.openManagedProject(ProjectRecord(id: UUID().uuidString, name: "Missing", path: root.appendingPathComponent("missing").path, updatedAt: Date()), section: "Run")
        XCTAssertEqual(store.activeSection, section)
        XCTAssertEqual(store.activeProjectID, project)
    }
    @MainActor func testCancelRetainsPriorPreviewUntilExitAndRejectsLateEvents() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let csv = root.appendingPathComponent("source.csv"); try "a,b\n1,2\n".write(to: csv, atomically: true, encoding: .utf8)
        let store = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        store.graph.execution = ExecutionSettings(datasetCSV: csv.path)
        store.dataPreviewSourcePath = csv.path; store.dataColumns = ["verified"]; store.dataRows = [["old"]]; store.dataRowCount = 1
        var exits: [(Int32, String, SourceInspectionResult?) -> Void] = [], hashes: [String] = [], cancelled = 0
        store.dataInspectionExecute = { _, project, _, _, source, hash, _, exit in
            XCTAssertNotNil(UUID(uuidString: project)); XCTAssertNotEqual(source.path, csv.path)
            exits.append(exit); hashes.append(hash)
            return .init(cancel: { cancelled += 1 })
        }
        store.inspectData(); XCTAssertTrue(store.isBusy)
        await waitUntil { exits.count == 1 }
        XCTAssertNotNil(store.dataInspectionHandle)
        store.inspectData(); XCTAssertEqual(exits.count, 1)
        store.cancelRun(); XCTAssertEqual(cancelled, 1)
        XCTAssertTrue(store.isBusy); XCTAssertEqual(store.dataColumns, ["verified"])
        exits[0](0, "", .init(columns: ["late"], rows: [["1"]], rowCount: 1, sourceSHA256: hashes[0]))
        await waitUntil { !store.isBusy }
        XCTAssertNil(store.dataInspectionHandle); XCTAssertEqual(store.dataColumns, ["verified"])
        store.inspectData(); await waitUntil { exits.count == 2 }
        exits[0](1, "obsolete failure", nil); XCTAssertTrue(store.isBusy)
        exits[1](0, "", .init(columns: ["a", "b"], rows: [["1", "2"]], rowCount: 1, sourceSHA256: hashes[1]))
        await waitUntil { !store.isBusy }
        XCTAssertEqual(store.dataColumns, ["a", "b"]); XCTAssertNil(store.activeProjectID)
        store.inspectData(); await waitUntil { exits.count == 3 }
        try "a,b\n3,4\n".write(to: csv, atomically: true, encoding: .utf8)
        exits[2](0, "", .init(columns: ["changed"], rows: [["1"]], rowCount: 1, sourceSHA256: hashes[2]))
        await waitUntil { !store.isBusy }
        XCTAssertEqual(store.dataColumns, ["a", "b"]); XCTAssertNotNil(store.dataError)
    }
    @MainActor func testNewSourceClearsOldColumnsBeforeCaptureAndFirstTerminalWins() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let csv = root.appendingPathComponent("new.csv"); try "a,b\n1,2\n".write(to: csv, atomically: true, encoding: .utf8)
        let store = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        store.graph.execution = ExecutionSettings(datasetCSV: csv.path)
        store.dataPreviewSourcePath = root.appendingPathComponent("old.csv").path
        store.dataColumns = ["old"]; store.dataRows = [["old"]]; store.dataRowCount = 1
        var terminalDelivered = false
        store.dataInspectionExecute = { _, _, _, _, _, hash, _, exit in
            exit(0, "", .init(columns: ["a", "b"], rows: [["1", "2"]], rowCount: 1, sourceSHA256: hash))
            XCTAssertTrue(store.isBusy)
            exit(1, "duplicate failure", nil)
            XCTAssertTrue(store.isBusy)
            terminalDelivered = true
            return .init(cancel: { XCTFail("Already exited worker must not be cancelled or attached.") })
        }
        store.inspectData()
        XCTAssertTrue(store.dataColumns.isEmpty); XCTAssertTrue(store.dataRows.isEmpty)
        XCTAssertEqual(store.dataRowCount, 0); XCTAssertNil(store.dataPreviewSourcePath)
        await waitUntil { terminalDelivered }
        XCTAssertNil(store.dataInspectionHandle)
        await waitUntil { !store.isBusy }
        XCTAssertEqual(store.dataColumns, ["a", "b"]); XCTAssertNil(store.dataError)
    }

    @MainActor private func waitUntil(_ ready: () -> Bool) async {
        let deadline = Date().addingTimeInterval(5)
        while !ready() && Date() < deadline { try? await Task.sleep(nanoseconds: 10_000_000) }
        XCTAssertTrue(ready())
    }

    @MainActor func testActualInspectionWorkerCanBeCancelledWithoutDiscardingPriorPreview() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let csv = root.appendingPathComponent("source.csv"); try "a,b\n1,2\n".write(to: csv, atomically: true, encoding: .utf8)
        let store = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        store.graph.execution = ExecutionSettings(datasetCSV: csv.path)
        store.dataPreviewSourcePath = csv.path; store.dataColumns = ["verified"]
        store.inspectData(); XCTAssertTrue(store.isInspectingData)
        store.cancelRun(); XCTAssertTrue(store.isBusy)
        let deadline = Date().addingTimeInterval(10)
        while store.isInspectingData && Date() < deadline { try await Task.sleep(nanoseconds: 25_000_000) }
        XCTAssertFalse(store.isBusy); XCTAssertNil(store.dataInspectionHandle)
        XCTAssertEqual(store.dataColumns, ["verified"])
        XCTAssertTrue(store.statusMessage.contains("cancelled"))
    }

}
