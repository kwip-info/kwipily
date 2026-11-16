import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DatabaseSourceProviderTests: XCTestCase {
    private let address = WorkflowNodeAddress(domain: .data, localID: "database")
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private struct FakeCredentials: DatabaseSourceCredentialStorage {
        func read(projectID: String, nodeID: String, connectionID: String) throws -> DatabaseSourceCredentials? { .init(username: "synthetic-user", password: "synthetic-password") }
        func write(_ value: DatabaseSourceCredentials, projectID: String, nodeID: String, connectionID: String) throws {}
        func clear(projectID: String, nodeID: String, connectionID: String) throws {}
    }
    @MainActor private struct Fixture {
        let root: URL; let workspace: WorkspaceStore; let owner: WorkflowAuthoringStore
        let bindings: WorkflowPrivateSourceBindings; let provider: WorkflowSourceProvider
    }
    @MainActor private func setup() throws -> Fixture {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        let library = ProjectLibrary(directory: root.appendingPathComponent("projects"))
        let options = try DatabaseSourceSettings.normalized(["backend": .string("postgresql"), "host": .string("db.example.test"),
            "database": .string("example"), "schema": .string("public"), "table": .string("observations")])
        let document = WorkflowDocument(projectID: UUID().uuidString, nodes: [.init(address: address,
            payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: ["format": .string("database"), "options": .object(options)]))])
        _ = try library.saveSource(.workflow(document))
        let owner = try WorkflowAuthoringStore(library: library, projectID: document.projectID, modelRegistry: .builtIn(),
            dataRegistry: .load(from: repo.appendingPathComponent("runner/dataflow-operators.json")))
        workspace.activeProjectID = owner.projectID
        XCTAssertTrue(owner.select(.init(nodeAddresses: [address]), lease: owner.lease()))
        let bindings = try WorkflowPrivateSourceBindings(owner: owner, directory: root)
        let provider = try WorkflowSourceProvider(workspace: workspace, owner: owner, bindings: bindings)
        provider.credentialStorage = FakeCredentials(); provider.rememberSource = { _ in }
        return .init(root: root, workspace: workspace, owner: owner, bindings: bindings, provider: provider)
    }
    @MainActor private func waitFor(_ predicate: () -> Bool) async throws {
        for _ in 0..<500 { if predicate() { return }; try await Task.sleep(nanoseconds: 20_000_000) }
        XCTFail("Database provider did not reach the expected state")
    }
    private static func captured(_ output: FileHandle) throws -> Data {
        let bytes = Data("bounded synthetic table".utf8)
        try output.write(contentsOf: bytes)
        return try JSONSerialization.data(withJSONObject: ["schemaVersion": 1, "kind": "databaseCapture", "format": "ipc",
            "sampleOnly": false, "bytes": bytes.count, "rows": 1, "columns": 1, "sha256": WorkflowArchiveCodec.hash(bytes)])
    }
    @MainActor func testAsynchronousFailureIsSourceScopedVisibleAndSanitizedWithoutBinding() async throws {
        let f = try setup(); defer { try? FileManager.default.removeItem(at: f.root) }
        var inspections = 0
        f.provider.inspectionExecute = { _, _, _, _, _, _, _, _ in inspections += 1; return WorkflowWorkerHandle(cancel: {}) }
        f.provider.databaseCaptureExecutor = { _, _, _ in throw NSError(domain: "synthetic", code: 1, userInfo: [NSLocalizedDescriptionKey: "password=do-not-show"]) }
        XCTAssertTrue(f.provider.connectDatabaseSource(for: address, lease: f.owner.lease()))
        XCTAssertEqual(f.provider.sourceInspectionStatus(for: address)?.phase, .inspecting)
        try await waitFor { !f.provider.isInspecting }
        let status = try XCTUnwrap(f.provider.sourceInspectionStatus(for: address))
        XCTAssertEqual(status.phase, .failed); XCTAssertTrue(status.isProblem)
        XCTAssertFalse(status.detail.contains("do-not-show")); XCTAssertNil(f.bindings.path(for: address)); XCTAssertEqual(inspections, 0)
        XCTAssertFalse(f.workspace.isPreparingData); XCTAssertFalse(f.owner.isBusy)
    }
    @MainActor func testOneConnectActionAutomaticallyInspectsAfterCaptureCommitAndBusyRelease() async throws {
        let f = try setup(); defer { try? FileManager.default.removeItem(at: f.root) }
        var exit: ((Int32, String, SourceInspectionResult?) -> Void)?, digest = ""
        f.provider.databaseCaptureExecutor = { _, _, output in try Self.captured(output) }
        f.provider.inspectionExecute = { _, _, _, _, _, hash, _, completion in
            digest = hash; exit = completion; return WorkflowWorkerHandle(cancel: {})
        }
        XCTAssertTrue(f.provider.connectDatabaseSource(for: address, lease: f.owner.lease()))
        try await waitFor { exit != nil }
        XCTAssertNotNil(f.bindings.path(for: address)); XCTAssertTrue(f.provider.isInspecting)
        XCTAssertEqual(f.provider.sourceInspectionStatus(for: address)?.phase, .inspecting)
        exit?(0, "", .init(columns: ["x"], rows: [["1"]], rowCount: 1, sourceSHA256: digest))
        try await waitFor { !f.provider.isInspecting }
        XCTAssertEqual(f.provider.sourceInspectionPreview(for: address)?.columns, ["x"])
        XCTAssertEqual(f.provider.sourceInspectionStatus(for: address)?.phase, .verified)
    }
    @MainActor func testCancelledConnectDoesNotCommitOrStartPreview() async throws {
        let f = try setup(); defer { try? FileManager.default.removeItem(at: f.root) }
        var entered = false, inspections = 0
        f.provider.databaseCaptureExecutor = { _, _, _ in entered = true; try await Task.sleep(nanoseconds: 30_000_000_000); throw CancellationError() }
        f.provider.inspectionExecute = { _, _, _, _, _, _, _, _ in inspections += 1; return WorkflowWorkerHandle(cancel: {}) }
        XCTAssertTrue(f.provider.connectDatabaseSource(for: address, lease: f.owner.lease()))
        try await waitFor { entered }; f.provider.cancel()
        try await waitFor { !f.provider.isInspecting }
        XCTAssertNil(f.bindings.path(for: address)); XCTAssertEqual(inspections, 0)
        XCTAssertEqual(f.provider.sourceInspectionStatus(for: address)?.phase, .cancelled)
    }
    @MainActor func testRetiredAuthorityCannotCommitOrAutoInspectDelayedCapture() async throws {
        let f = try setup(); defer { try? FileManager.default.removeItem(at: f.root) }
        var continuation: CheckedContinuation<Void, Never>?, inspections = 0, remembered = 0
        f.provider.rememberSource = { _ in remembered += 1 }
        f.provider.databaseCaptureExecutor = { _, _, output in
            await withCheckedContinuation { continuation = $0 }
            return try Self.captured(output)
        }
        f.provider.inspectionExecute = { _, _, _, _, _, _, _, _ in inspections += 1; return WorkflowWorkerHandle(cancel: {}) }
        XCTAssertTrue(f.provider.connectDatabaseSource(for: address, lease: f.owner.lease()))
        try await waitFor { continuation != nil }; f.owner.invalidate(); continuation?.resume()
        try await waitFor { !f.provider.isInspecting }
        XCTAssertNil(f.bindings.path(for: address)); XCTAssertEqual(inspections, 0); XCTAssertEqual(remembered, 0)
        XCTAssertNil(f.provider.sourceInspectionStatus(for: address))
    }
    @MainActor func testCapturedBindingCommitRequiresExactBusyReservationLeaseAndRevision() throws {
        let f = try setup(); defer { try? FileManager.default.removeItem(at: f.root) }
        let path = f.root.appendingPathComponent("private-capture.arrow").path
        let lease = f.owner.lease(), revision = f.bindings.revision
        f.workspace.cancelDataPreparation = {}
        let reservation = f.workspace.dataPreparationReservationID
        f.workspace.isPreparingData = true; f.owner.setExecutionBusy(true)
        defer { f.workspace.isPreparingData = false; f.workspace.cancelDataPreparation = nil; f.owner.setExecutionBusy(false) }
        XCTAssertFalse(f.bindings.setPath(path, for: address, lease: lease, expectedRevision: revision), "Ordinary source edits must remain forbidden while capture is active.")
        XCTAssertFalse(f.bindings.commitCapturedPath(path, for: address, lease: lease, expectedRevision: revision,
            workspace: f.workspace, reservationID: UUID()))
        let stale = WorkflowEditorLease(projectID: lease.projectID, sessionID: lease.sessionID, generation: lease.generation + 1)
        XCTAssertFalse(f.bindings.commitCapturedPath(path, for: address, lease: stale, expectedRevision: revision,
            workspace: f.workspace, reservationID: reservation))
        XCTAssertFalse(f.bindings.commitCapturedPath(path, for: address, lease: lease, expectedRevision: revision + 1,
            workspace: f.workspace, reservationID: reservation))
        XCTAssertNil(f.bindings.path(for: address))
        XCTAssertTrue(f.bindings.commitCapturedPath(path, for: address, lease: lease, expectedRevision: revision,
            workspace: f.workspace, reservationID: reservation))
        XCTAssertTrue(f.owner.isBusy); XCTAssertTrue(f.workspace.isPreparingData)
        XCTAssertEqual(f.bindings.path(for: address), path)
        f.workspace.cancelDataPreparation = {}
        XCTAssertFalse(f.bindings.commitCapturedPath(f.root.appendingPathComponent("replacement.arrow").path, for: address,
            lease: lease, expectedRevision: f.bindings.revision, workspace: f.workspace, reservationID: reservation),
            "A replaced cancellation owner rotates the reservation and revokes the old capture.")
        XCTAssertEqual(f.bindings.path(for: address), path)
    }

}
