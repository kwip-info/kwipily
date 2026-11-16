import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowPreparationInspectionSessionTests: XCTestCase {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func fixture() throws -> WorkflowPreparationInspectionInput {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return .init(reference: .init(projectID: UUID().uuidString, directoryID: UUID()), runID: UUID().uuidString,
            requestIdentity: String(repeating: "a", count: 64), terminalStatus: .completed, capturedRevision: 1,
            directory: directory, manifestSHA256: String(repeating: "b", count: 64), graphDigest: String(repeating: "c", count: 64),
            registryDigest: String(repeating: "d", count: 64), numpyVersion: "2.5.3", graph: .init(nodes: [], edges: []), nodes: [],
            cacheReport: .init(policy: "off", executedNodeIDs: [], reusedNodeIDs: [], nodes: []))
    }
    private func runtime() throws -> RuntimeLocation { try RuntimeLocation.resolve(resources: nil, developmentRoot: repo, allowDevelopment: true) }
    func testCancelDuringStagingDiscardsLateCopyAndReleasesOnlyAfterStagingReturns() async throws {
        let input = try fixture(), stager = PreparationStagerProbe(), token = UUID()
        var released: [UUID] = [], launches = 0
        let session = WorkflowPreparationInspectionSession(reference: input.reference, root: repo, runtime: try runtime(),
            reservation: .init(acquire: { token }, release: { released.append($0) }), stager: stager,
            execute: { _, _, _, _, _, _, _ in launches += 1; return .init(cancel: {}) })
        XCTAssertTrue(session.start())
        await stager.waitForRequest()
        session.cancel()
        XCTAssertTrue(session.isBusy); XCTAssertTrue(released.isEmpty)
        await stager.finish(input)
        await idle(session)
        XCTAssertEqual(released, [token]); XCTAssertEqual(launches, 0); XCTAssertNil(session.snapshot)
        XCTAssertFalse(FileManager.default.fileExists(atPath: input.directory.path))
        XCTAssertFalse(session.start(), "An accepted operation cannot be restarted.")
    }
    func testCancelWorkerRetainsStagingUntilExitAndIgnoresDuplicateLateExit() async throws {
        let input = try fixture(), stager = PreparationStagerProbe(), token = UUID()
        var released: [UUID] = [], cancels = 0
        var exit: ((Int32, String, PreparationInspectionResult?) -> Void)?
        let launched = expectation(description: "Worker launched")
        let session = WorkflowPreparationInspectionSession(reference: input.reference, root: repo, runtime: try runtime(),
            reservation: .init(acquire: { token }, release: { released.append($0) }), stager: stager,
            execute: { _, _, _, _, _, _, completion in exit = completion; launched.fulfill(); return .init(cancel: { cancels += 1 }) })
        XCTAssertTrue(session.start()); await stager.waitForRequest(); await stager.finish(input)
        await fulfillment(of: [launched], timeout: 3)
        session.cancel()
        XCTAssertEqual(cancels, 1); XCTAssertTrue(session.isBusy); XCTAssertTrue(released.isEmpty)
        XCTAssertTrue(FileManager.default.fileExists(atPath: input.directory.path))
        exit?(0, "", nil); await idle(session)
        XCTAssertEqual(released, [token]); XCTAssertNil(session.snapshot)
        XCTAssertFalse(FileManager.default.fileExists(atPath: input.directory.path))
        exit?(0, "", nil)
        XCTAssertEqual(released, [token]); XCTAssertNil(session.snapshot)
    }
    func testSynchronousFailureDoesNotInstallDeadWorkerAndRefusedReservationDoesNotStage() async throws {
        let input = try fixture(), stager = PreparationStagerProbe()
        var releases = 0, cancels = 0
        let session = WorkflowPreparationInspectionSession(reference: input.reference, root: repo, runtime: try runtime(),
            reservation: .init(acquire: { UUID() }, release: { _ in releases += 1 }), stager: stager,
            execute: { _, _, _, _, _, _, exit in exit(1, "Unsupported historical registry", nil); return .init(cancel: { cancels += 1 }) })
        XCTAssertTrue(session.start()); await stager.waitForRequest(); await stager.finish(input); await idle(session)
        XCTAssertEqual(releases, 1); XCTAssertEqual(cancels, 0)
        XCTAssertTrue(session.statusMessage.contains("Unsupported historical registry"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: input.directory.path))
        let refused = WorkflowPreparationInspectionSession(reference: input.reference, root: repo, runtime: try runtime(),
            reservation: .init(acquire: { nil }, release: { _ in XCTFail("No reservation owned") }), stager: stager)
        XCTAssertFalse(refused.start()); XCTAssertFalse(refused.isBusy)
    }
    func testChangedSelectionAtCompletionPreservesPreviousSnapshotAndReleasesReservation() async throws {
        let input = try fixture(), stager = PreparationStagerProbe()
        let previous = WorkflowPreparationInspectionSnapshot(input: input, outputs: [:], cacheReport: input.cacheReport)
        var accepted = true, released = 0
        var finish: ((Int32, String, PreparationInspectionResult?) -> Void)?
        let launched = expectation(description: "Typed inspection accepted")
        let session = WorkflowPreparationInspectionSession(reference: input.reference, root: repo, runtime: try runtime(),
            reservation: .init(acquire: { UUID() }, release: { _ in released += 1 }), stager: stager,
            execute: { _, _, _, _, _, _, exit in finish = exit; launched.fulfill(); return .init(cancel: {}) },
            previous: previous, accepts: { _ in accepted })
        XCTAssertTrue(session.start()); await stager.waitForRequest(); await stager.finish(input)
        await fulfillment(of: [launched], timeout: 3)
        accepted = false
        finish?(0, "", .init(outputs: [:], cacheReport: input.cacheReport))
        await idle(session)
        XCTAssertEqual(released, 1)
        XCTAssertNotNil(session.snapshot)
        XCTAssertTrue(session.statusMessage.contains("changed"), session.statusMessage)
        XCTAssertFalse(FileManager.default.fileExists(atPath: input.directory.path))
    }
    func testTimeoutPreservesPreviousSnapshotAndReportsBudgetWithoutDeadWorker() async throws {
        let input = try fixture(), stager = PreparationStagerProbe()
        let previous = WorkflowPreparationInspectionSnapshot(input: input, outputs: [:], cacheReport: input.cacheReport)
        var releases = 0, cancels = 0
        let session = WorkflowPreparationInspectionSession(reference: input.reference, root: repo, runtime: try runtime(),
            reservation: .init(acquire: { UUID() }, release: { _ in releases += 1 }), stager: stager,
            execute: { _, _, _, window, _, _, exit in
                XCTAssertEqual(window.policy.maximumRunSeconds, 3600)
                exit(124, "Time limit", nil); return .init(cancel: { cancels += 1 })
            }, previous: previous)
        XCTAssertTrue(session.start()); await stager.waitForRequest(); await stager.finish(input); await idle(session)
        XCTAssertEqual(releases, 1); XCTAssertEqual(cancels, 0); XCTAssertNotNil(session.snapshot)
        XCTAssertTrue(session.statusMessage.contains("time allowance"), session.statusMessage)
    }
    private func idle(_ session: WorkflowPreparationInspectionSession) async {
        if !session.isBusy { return }
        let done = expectation(description: "Inspection terminates")
        let observation = session.$isBusy.filter { !$0 }.prefix(1).sink { _ in done.fulfill() }
        await fulfillment(of: [done], timeout: 3); observation.cancel()
    }
}

private actor PreparationStagerProbe: WorkflowPreparationStaging {
    private var pending: CheckedContinuation<WorkflowPreparationInspectionInput, Error>?
    private var waiter: CheckedContinuation<Void, Never>?
    func stage(_ reference: WorkflowHistoryReference, root: URL, window: ExecutionBudgetWindow) async throws -> WorkflowPreparationInspectionInput {
        try await withCheckedThrowingContinuation { pending = $0; waiter?.resume(); waiter = nil }
    }
    func waitForRequest() async { if pending == nil { await withCheckedContinuation { waiter = $0 } } }
    func finish(_ input: WorkflowPreparationInspectionInput) { pending?.resume(returning: input); pending = nil }
}
