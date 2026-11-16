import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowExecutionStorageAdmissionTests: XCTestCase {
    func testExactCapturedCountsAndRetainedBytesSurvivePhaseReadmission() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let project = UUID().uuidString, run = UUID().uuidString
        let admission = try WorkflowExecutionStorageAdmission(workspace: root, projectID: project, directoryID: run,
            preparationOperationID: UUID().uuidString, trainingOperationID: UUID().uuidString,
            sourceByteCounts: [20], checkpointFiles: [], limitBytes: 669_000_020)
        let ledger = try WorkflowStorageAdmission(workspace: root)
        XCTAssertEqual(try ledger.snapshot().committedBytes, 669_000_020)
        let directory = root.appendingPathComponent("workflow-runs/\(project)/\(run)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(repeating: 1, count: 10).write(to: directory.appendingPathComponent("retained"))
        try admission.transitionToTraining()
        XCTAssertEqual(try ledger.snapshot().committedBytes, 669_000_020)
        admission.close()
        XCTAssertEqual(try ledger.snapshot().committedBytes, 10)
        for counts in [[-1], [10_000_001], Array(repeating: 10_000_000, count: 26)] {
            XCTAssertThrowsError(try WorkflowExecutionStorageAdmission(workspace: root, projectID: project, directoryID: UUID().uuidString,
                preparationOperationID: UUID().uuidString, trainingOperationID: UUID().uuidString,
                sourceByteCounts: counts, checkpointFiles: [], limitBytes: Int64.max))
        }
    }
    func testCollectionAllowanceIsExplicitAndIncludesEveryMemberByte() throws {
        let count = FileInputCollectionManifest.maximumMemberBytes + 1000
        XCTAssertThrowsError(try WorkflowExecutionStorageAdmission.peakBytes(sourceByteCounts: [count], checkpointFiles: []))
        XCTAssertEqual(try WorkflowExecutionStorageAdmission.peakBytes(sourceByteCounts: [count], checkpointFiles: [], hasFileCollections: true), 669_000_000 + Int64(count))
        for counts in [[-1], [WorkflowCapturedSourceArchive.maximumAggregateBytes + 1], [WorkflowCapturedSourceArchive.maximumAggregateBytes, 1]] {
            XCTAssertThrowsError(try WorkflowExecutionStorageAdmission.peakBytes(sourceByteCounts: counts, checkpointFiles: [], hasFileCollections: true))
        }
    }
    @MainActor func testSessionRejectsUnsupervisedQuotaBeforeArchiveCreation() throws {
        let (root, owner, runtime) = try WorkflowExecutionRequestTests().setup()
        defer { try? FileManager.default.removeItem(at: root) }
        let request = try WorkflowExecutionRequest.capture(from: owner, train: .init(domain: .control, localID: "train"), runtime: runtime)
        let source = root.appendingPathComponent("source.csv")
        try Data("a,b,label\n1,2,0\n2,3,1\n".utf8).write(to: source)
        let directory = root.appendingPathComponent("workflow-runs/\(owner.projectID)/\(UUID().uuidString)")
        let session = WorkflowExecutionSession(request: request, owner: owner,
            sources: [.init(domain: .data, localID: "input"): try .capture(source)], directory: directory,
            reservation: .init(acquire: { UUID() }, release: { _ in }), storageLimitBytes: Int64.max,
            launch: { _, _, _, _, _ in XCTFail("Unsupervised request must not launch"); return .init(cancel: {}) }, onTerminal: { _ in })
        XCTAssertFalse(session.start())
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
    }
    @MainActor func testAdmissionRefusesBeforeWritesAndCancellationWaitsForActualExit() async throws {
        for refuse in [true, false] {
            let (root, owner, runtime) = try WorkflowExecutionRequestTests().setup()
            defer { try? FileManager.default.removeItem(at: root) }
            let policy = try ResolvedExecutionBudget.resolve(global: .init(maximumRunSeconds: 30), projectMaximumRunSeconds: nil)
            let request = try WorkflowExecutionRequest.capture(from: owner, train: .init(domain: .control, localID: "train"), runtime: runtime, executionBudget: policy)
            let source = root.appendingPathComponent("source.csv")
            try Data("a,b,label\n1,2,0\n2,3,1\n".utf8).write(to: source)
            let captured = try WorkflowCapturedSource.capture(source), directory = root.appendingPathComponent("workflow-runs/\(owner.projectID)/\(UUID().uuidString)")
            let terminal = expectation(description: "Terminal \(refuse)"), launched = expectation(description: "Launched")
            var exit: ((Int32, String) -> Void)?, cancellations = 0, releases = 0, outcome: WorkflowExecutionOutcome?
            let session = WorkflowExecutionSession(request: request, owner: owner,
                sources: [.init(domain: .data, localID: "input"): captured], directory: directory,
                reservation: .init(acquire: { UUID() }, release: { _ in releases += 1 }),
                storageLimitBytes: refuse ? 0 : Int64(669_000_000 + captured.byteCount),
                launch: { _, _, _, _, callback in exit = callback; launched.fulfill(); return .init(cancel: { cancellations += 1 }) },
                onTerminal: { outcome = $0; terminal.fulfill() })
            XCTAssertTrue(session.start())
            if refuse {
                launched.isInverted = true
                await fulfillment(of: [terminal, launched], timeout: 0.5)
                XCTAssertEqual(outcome?.status, .failed)
                XCTAssertFalse(FileManager.default.fileExists(atPath: directory.path))
            } else {
                await fulfillment(of: [launched], timeout: 5)
                let ledger = try WorkflowStorageAdmission(workspace: root)
                XCTAssertEqual(try ledger.snapshot().committedBytes, Int64(669_000_000 + captured.byteCount))
                session.cancel()
                XCTAssertEqual(cancellations, 1); XCTAssertEqual(releases, 0)
                XCTAssertEqual(try ledger.snapshot().committedBytes, Int64(669_000_000 + captured.byteCount))
                exit?(143, "")
                await fulfillment(of: [terminal], timeout: 5)
                XCTAssertEqual(outcome?.status, .cancelled)
                XCTAssertGreaterThan(try ledger.snapshot().retainedBytes, 0)
                XCTAssertEqual(try ledger.snapshot().reservedGrowthBytes, 0)
            }
            XCTAssertEqual(releases, 1)
        }
    }
    @MainActor func testRealPhaseTransitionCompletesOrPreservesPreparationWhenReadmissionFails() async throws {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else { throw XCTSkip("Requires actual runtime") }
        for refuseTraining in [false, true] {
            let (root, owner, _) = try WorkflowExecutionRequestTests().setup()
            defer { try? FileManager.default.removeItem(at: root) }
            let runtime = try RuntimeLocation.resolve(resources: nil, developmentRoot: URL(fileURLWithPath: development), allowDevelopment: true)
            let policy = try ResolvedExecutionBudget.resolve(global: .init(maximumRunSeconds: 60), projectMaximumRunSeconds: nil)
            let request = try WorkflowExecutionRequest.capture(from: owner, train: .init(domain: .control, localID: "train"), runtime: runtime, executionBudget: policy)
            let source = root.appendingPathComponent("source.csv")
            let rows = (0..<40).map { "\($0),\($0 + 1),\($0 % 2)" }.joined(separator: "\n")
            try Data(("a,b,label\n" + rows + "\n").utf8).write(to: source)
            let captured = try WorkflowCapturedSource.capture(source)
            let directory = root.appendingPathComponent("workflow-runs/\(owner.projectID)/\(UUID().uuidString)")
            let terminal = expectation(description: "Real storage run")
            var outcome: WorkflowExecutionOutcome?, phases: [WorkflowLaunchIntent.Phase] = [], operationIDs: [String] = []
            let session = WorkflowExecutionSession(request: request, owner: owner,
                sources: [.init(domain: .data, localID: "input"): captured], directory: directory,
                reservation: .init(acquire: { UUID() }, release: { _ in }), storageLimitBytes: Int64(669_000_000 + captured.byteCount),
                launch: { executable, args, cwd, line, exit in
                    let index = try XCTUnwrap(args.firstIndex(of: "--kwip-operation-supervise"))
                    let operation = URL(fileURLWithPath: args[index + 1])
                    let envelope = try WorkflowOperationRequest.decode(Data(contentsOf: operation.appendingPathComponent("request.json")))
                    phases.append(envelope.phase); operationIDs.append(envelope.operationID)
                    return try WorkflowExecutionSession.launchWorker(executable, args, cwd, line, { code, error in
                        if envelope.phase == .preparation, refuseTraining, code == 0 {
                            do {
                                let other = root.appendingPathComponent("runs/" + UUID().uuidString)
                                try FileManager.default.createDirectory(at: other, withIntermediateDirectories: true)
                                try Data(repeating: 1, count: 100).write(to: other.appendingPathComponent("retained"))
                            } catch { XCTFail(error.localizedDescription) }
                        }
                        exit(code, error)
                    })
                }, onTerminal: { outcome = $0; terminal.fulfill() })
            XCTAssertTrue(session.start())
            await fulfillment(of: [terminal], timeout: 60)
            XCTAssertEqual(outcome?.status, refuseTraining ? .failed : .completed, outcome?.error ?? "No terminal outcome")
            XCTAssertEqual(phases, refuseTraining ? [.preparation] : [.preparation, .training])
            XCTAssertEqual(Set(operationIDs).count, phases.count)
            XCTAssertTrue(FileManager.default.fileExists(atPath: directory.appendingPathComponent("preparation/artifacts/manifest.json").path))
            if refuseTraining { XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appendingPathComponent("training").path)) }
            XCTAssertEqual(try WorkflowStorageAdmission(workspace: root).snapshot().reservedGrowthBytes, 0)
        }
    }

}
