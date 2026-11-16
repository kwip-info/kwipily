import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowTrainedExportStoreTests: XCTestCase {
    private func package() throws -> WorkflowTrainedExportPackage {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: path, withIntermediateDirectories: false)
        addTeardownBlock { try? FileManager.default.removeItem(at: path) }
        return .init(reference: .init(projectID: UUID().uuidString, directoryID: UUID()), runID: UUID().uuidString,
            requestIdentity: String(repeating: "a", count: 64), modelManifestSHA256: String(repeating: "b", count: 64),
            directory: path, files: [], manifestBytes: Data(),
            inputContract: try TrainedModelInputContract(graph:ExampleFlows.parallelClassifier,registry:.builtIn(),columns:["x","y"],target:"label"))
    }
    func testCancelledStagingCannotPublishLateReviewAndCleansPrivateCopy() async throws {
        let package = try package(), io = ExportIOProbe()
        var releases = 0
        let store = WorkflowTrainedExportStore(reference: package.reference, root: package.directory,
            runID: package.runID, requestIdentity: package.requestIdentity, modelManifestSHA256: package.modelManifestSHA256,
            reservation: .init(acquire: { UUID() }, release: { _ in releases += 1 }), io: io, isCurrent: { true })
        XCTAssertTrue(store.prepare()); await io.waitForStage()
        store.cancel(); XCTAssertTrue(store.isBusy); XCTAssertEqual(releases, 0)
        await io.finishStage(package); await idle(store)
        XCTAssertNil(store.package); XCTAssertEqual(releases, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: package.directory.path))
    }
    func testSaveOwnsReviewFilesUntilExitAndInvalidationCannotPublishNewSelection() async throws {
        let package = try package(), io = ExportIOProbe()
        var releases = 0
        let store = WorkflowTrainedExportStore(reference: package.reference, root: package.directory,
            runID: package.runID, requestIdentity: package.requestIdentity, modelManifestSHA256: package.modelManifestSHA256,
            reservation: .init(acquire: { UUID() }, release: { _ in releases += 1 }), io: io, isCurrent: { true })
        XCTAssertTrue(store.prepare()); await io.waitForStage(); await io.finishStage(package); await idle(store)
        XCTAssertNotNil(store.package); XCTAssertEqual(releases, 1)
        XCTAssertTrue(store.save(to: package.directory.appendingPathComponent("saved")))
        await io.waitForSave(); store.invalidate()
        XCTAssertTrue(FileManager.default.fileExists(atPath: package.directory.path)); XCTAssertTrue(store.isBusy)
        await io.failSave(); await idle(store)
        XCTAssertNil(store.package); XCTAssertNil(store.savedURL); XCTAssertEqual(releases, 2)
        XCTAssertFalse(FileManager.default.fileExists(atPath: package.directory.path))
        XCTAssertFalse(store.prepare())
    }
    func testChangedSelectedIdentityRejectsStagingAndReadyInvalidationCleans() async throws {
        let package = try package(), io = ExportIOProbe()
        var current = true
        let store = WorkflowTrainedExportStore(reference: package.reference, root: package.directory,
            runID: package.runID, requestIdentity: "different", modelManifestSHA256: package.modelManifestSHA256,
            reservation: .init(acquire: { UUID() }, release: { _ in }), io: io, isCurrent: { current })
        XCTAssertTrue(store.prepare()); await io.waitForStage(); await io.finishStage(package); await idle(store)
        XCTAssertNil(store.package); XCTAssertFalse(FileManager.default.fileExists(atPath: package.directory.path))
        current = false
        XCTAssertFalse(store.prepare()); XCTAssertFalse(store.save(to: package.directory))
    }
    func testCommittedSaveRemainsSuccessfulWhenCancellationArrivesAfterCommit() async throws {
        let package = try package(), io = ExportIOProbe()
        var releases = 0
        let store = WorkflowTrainedExportStore(reference: package.reference, root: package.directory,
            runID: package.runID, requestIdentity: package.requestIdentity, modelManifestSHA256: package.modelManifestSHA256,
            reservation: .init(acquire: { UUID() }, release: { _ in releases += 1 }), io: io, isCurrent: { true })
        XCTAssertTrue(store.prepare()); await io.waitForStage(); await io.finishStage(package); await idle(store)
        let destination = package.directory.deletingLastPathComponent().appendingPathComponent("export")
        XCTAssertTrue(store.save(to: destination)); await io.waitForSave()
        store.cancel(); await io.finishSave(); await idle(store)
        XCTAssertEqual(store.savedURL, destination); XCTAssertEqual(releases, 2)
        XCTAssertNil(store.package)
        XCTAssertFalse(FileManager.default.fileExists(atPath: package.directory.path))
    }
    func testOrdinarySaveFailurePreservesReviewForRetryAndReadyInvalidationCleans() async throws {
        let package = try package(), io = ExportIOProbe()
        let store = WorkflowTrainedExportStore(reference: package.reference, root: package.directory,
            runID: package.runID, requestIdentity: package.requestIdentity, modelManifestSHA256: package.modelManifestSHA256,
            reservation: .init(acquire: { UUID() }, release: { _ in }), io: io, isCurrent: { true })
        XCTAssertTrue(store.prepare()); await io.waitForStage(); await io.finishStage(package); await idle(store)
        XCTAssertTrue(store.save(to: package.directory)); await io.waitForSave()
        XCTAssertFalse(store.prepare()); XCTAssertFalse(store.save(to: package.directory))
        await io.failSave(WorkflowArchiveError("Destination already exists.")); await idle(store)
        XCTAssertNotNil(store.package); XCTAssertNil(store.savedURL)
        XCTAssertTrue(FileManager.default.fileExists(atPath: package.directory.path))
        XCTAssertTrue(store.save(to: package.directory)); await io.waitForSave()
        await io.finishSave(); await idle(store)
        XCTAssertEqual(store.savedURL, package.directory)
        store.invalidate()
        XCTAssertNil(store.package); XCTAssertFalse(FileManager.default.fileExists(atPath: package.directory.path))
    }
    func testReservationRefusalAndChangedSelectionPreventSavingReview() async throws {
        let package = try package(), io = ExportIOProbe()
        var available = false, current = true, releases = 0
        let store = WorkflowTrainedExportStore(reference: package.reference, root: package.directory,
            runID: package.runID, requestIdentity: package.requestIdentity, modelManifestSHA256: package.modelManifestSHA256,
            reservation: .init(acquire: { available ? UUID() : nil }, release: { _ in releases += 1 }),
            io: io, isCurrent: { current })
        XCTAssertFalse(store.prepare()); XCTAssertFalse(store.isBusy); XCTAssertEqual(releases, 0)
        available = true
        XCTAssertTrue(store.prepare()); await io.waitForStage(); await io.finishStage(package); await idle(store)
        available = false
        XCTAssertFalse(store.save(to: package.directory)); XCTAssertNotNil(store.package)
        available = true; current = false
        XCTAssertFalse(store.save(to: package.directory)); XCTAssertEqual(releases, 1)
        store.invalidate()
    }
    func testClosingReadyReviewReleasesPrivateFilesWithoutExplicitCancellation() async throws {
        let package = try package(), io = ExportIOProbe()
        var store: WorkflowTrainedExportStore? = WorkflowTrainedExportStore(reference: package.reference, root: package.directory,
            runID: package.runID, requestIdentity: package.requestIdentity, modelManifestSHA256: package.modelManifestSHA256,
            reservation: .init(acquire: { UUID() }, release: { _ in }), io: io, isCurrent: { true })
        XCTAssertTrue(store!.prepare()); await io.waitForStage(); await io.finishStage(package); await idle(store!)
        weak var released = store
        store = nil
        // The staging task can still be returning after its final published state.
        for _ in 0..<20 where released != nil { await Task.yield() }
        XCTAssertNil(released)
        XCTAssertFalse(FileManager.default.fileExists(atPath: package.directory.path))
    }
    private func idle(_ store: WorkflowTrainedExportStore) async {
        if !store.isBusy { return }
        let done = expectation(description: "Export operation finishes")
        let observation = store.$isBusy.filter { !$0 }.prefix(1).sink { _ in done.fulfill() }
        await fulfillment(of: [done], timeout: 3); observation.cancel()
    }
}
private actor ExportIOProbe: WorkflowTrainedExportIO {
    private var staging: CheckedContinuation<WorkflowTrainedExportPackage, Error>?
    private var saving: CheckedContinuation<Void, Error>?
    private var stageWaiter: CheckedContinuation<Void, Never>?, saveWaiter: CheckedContinuation<Void, Never>?
    func stage(_ reference: WorkflowHistoryReference, root: URL) async throws -> WorkflowTrainedExportPackage {
        try await withCheckedThrowingContinuation { staging = $0; stageWaiter?.resume(); stageWaiter = nil }
    }
    func publish(_ package: WorkflowTrainedExportPackage, to destination: URL) async throws {
        try await withCheckedThrowingContinuation { saving = $0; saveWaiter?.resume(); saveWaiter = nil }
    }
    func waitForStage() async { if staging == nil { await withCheckedContinuation { stageWaiter = $0 } } }
    func waitForSave() async { if saving == nil { await withCheckedContinuation { saveWaiter = $0 } } }
    func finishStage(_ package: WorkflowTrainedExportPackage) { staging?.resume(returning: package); staging = nil }
    func failSave(_ error: Error = CancellationError()) { saving?.resume(throwing: error); saving = nil }
    func finishSave() { saving?.resume(); saving = nil }
}
