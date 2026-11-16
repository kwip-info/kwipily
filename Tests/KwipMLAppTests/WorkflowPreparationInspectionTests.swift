import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

final class WorkflowPreparationInspectionTests: XCTestCase {
    @MainActor func testActualArchivedInspectionUsesStagedBytesAndRejectsIncompleteOrChangedEvidence() async throws {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {
            throw XCTSkip("Enable actual workflow runtime for archived preparation inspection.")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: URL(fileURLWithPath: development), storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadPreparedWorkflowExample(device: "cpu"))
        let host = try XCTUnwrap(workspace.workflowSession)
        let terminal = expectation(description: "Actual workflow completes")
        let subscription = host.$outcomes.dropFirst().filter { !$0.isEmpty }.prefix(1).sink { _ in terminal.fulfill() }
        defer { subscription.cancel() }
        XCTAssertTrue(host.run(train: WorkflowExamples.trainAddress))
        await fulfillment(of: [terminal], timeout: 60)
        let outcome = try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(outcome.status, .completed)
        let reference = WorkflowHistoryReference(projectID: host.owner.projectID, directoryID: try XCTUnwrap(UUID(uuidString: outcome.directory.lastPathComponent)))
        XCTAssertTrue(host.bindings.setPath(nil, for: WorkflowExamples.sourceAddress,
            lease: host.owner.lease(), expectedRevision: host.bindings.revision))
        XCTAssertNotNil(host.owner.apply(.rename(address: WorkflowExamples.sourceAddress, name: "Current renamed source"), lease: host.owner.lease()))
        let history = WorkflowHistoryStore(workspace: workspace, host: host)
        history.select(reference)
        let selected = expectation(description: "History selection verified")
        let selectedObservation = history.$isLoading.filter { !$0 }.prefix(1).sink { _ in selected.fulfill() }
        await fulfillment(of: [selected], timeout: 10); selectedObservation.cancel()
        XCTAssertTrue(workspace.setExecutionPreferences(try .init(managedStorageLimitBytes: 1)))
        XCTAssertTrue(history.inspectPreparation())
        let refused = try XCTUnwrap(history.preparationInspection)
        let refusalDeadline = Date().addingTimeInterval(10)
        while refused.isBusy && Date() < refusalDeadline { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertNil(refused.snapshot)
        XCTAssertTrue(refused.statusMessage.lowercased().contains("storage"), refused.statusMessage)
        XCTAssertTrue(workspace.resetExecutionPreferences())
        XCTAssertTrue(history.inspectPreparation())
        let inspection = try XCTUnwrap(history.preparationInspection)
        let inspected = expectation(description: "History preparation displayed")
        let inspectionObservation = inspection.$isBusy.filter { !$0 }.prefix(1).sink { _ in inspected.fulfill() }
        await fulfillment(of: [inspected], timeout: 30); inspectionObservation.cancel()
        XCTAssertNotNil(inspection.snapshot, inspection.statusMessage)
        XCTAssertFalse(workspace.isBusy)
        var delayedExit: ((Int32, String, PreparationInspectionResult?) -> Void)?
        var cancelledWorkers = 0
        let launched = expectation(description: "Inspection worker held for selection test")
        XCTAssertTrue(workspace.setProjectExecutionOverrides(try .init(maximumRunSeconds: 80, cpuThreads: 3), projectID: host.owner.projectID))
        history.preparationExecute = { _, _, _, window, _, _, exit in
            XCTAssertEqual(window.policy.maximumRunSeconds, 80)
            XCTAssertEqual(window.policy.resources?.cpuThreads, 3)
            delayedExit = exit; launched.fulfill(); return .init(cancel: { cancelledWorkers += 1 })
        }
        XCTAssertTrue(history.inspectPreparation())
        await fulfillment(of: [launched], timeout: 10)
        let abandoned = try XCTUnwrap(history.preparationInspection)
        XCTAssertNotNil(abandoned.snapshot, "Refreshing retains the prior displayed snapshot until replacement succeeds.")
        history.select(reference)
        XCTAssertEqual(cancelledWorkers, 1)
        XCTAssertTrue(workspace.isPreparingData, "Worker still owns its reservation until exit.")
        workspace.cancelDataPreparation = {}
        let foreignReservation = workspace.dataPreparationReservationID
        delayedExit?(0, "", nil)
        XCTAssertTrue(workspace.isPreparingData)
        XCTAssertEqual(workspace.dataPreparationReservationID, foreignReservation)
        XCTAssertFalse(abandoned.isBusy)
        workspace.isPreparingData = false; workspace.cancelDataPreparation = nil
        XCTAssertNil(history.preparationInspection, "Changing selected evidence must detach previous inspection.")
        history.invalidate()
        let staging = root.appendingPathComponent("preparation-inspection-staging")
        let beforeRefusal = (try? FileManager.default.contentsOfDirectory(atPath: staging.path)) ?? []
        XCTAssertThrowsError(try WorkflowHistoryReader.stagePreparationInspection(reference, root: root, storageLimitBytes: 1))
        XCTAssertEqual((try? FileManager.default.contentsOfDirectory(atPath: staging.path)) ?? [], beforeRefusal,
                       "Admission failure must not write a first snapshot.")
        var releasedInspectionReservations = 0
        let budgetedSession = WorkflowPreparationInspectionSession(reference: reference, root: root,
            runtime: try XCTUnwrap(workspace.runtime),
            reservation: .init(acquire: { UUID() }, release: { _ in releasedInspectionReservations += 1 }),
            storageLimitBytes: 2_000_000_000)
        XCTAssertTrue(budgetedSession.start())
        let budgetedDone = expectation(description: "Admitted saved-output inspection completes")
        let budgetedObservation = budgetedSession.$isBusy.filter { !$0 }.prefix(1).sink { _ in budgetedDone.fulfill() }
        await fulfillment(of: [budgetedDone], timeout: 30); budgetedObservation.cancel()
        let budgetedSnapshot = try XCTUnwrap(budgetedSession.snapshot, budgetedSession.statusMessage)
        XCTAssertEqual(releasedInspectionReservations, 1)
        XCTAssertNil(budgetedSnapshot.input.storageReservation, "Displayed metadata must not retain a storage admission.")
        XCTAssertFalse(FileManager.default.fileExists(atPath: budgetedSnapshot.input.directory.path))
        let input = try WorkflowHistoryReader.stagePreparationInspection(reference, root: root, storageLimitBytes: 2_000_000_000)
        defer { input.remove() }
        XCTAssertNotNil(input.storageReservation)
        XCTAssertTrue(input.directory.path.contains("/preparation-inspection-staging/"))
        XCTAssertTrue(input.directory.path.hasSuffix("/input/preparation"))
        XCTAssertEqual(input.reference, reference)
        XCTAssertEqual(input.nodes.count, 6)
        XCTAssertEqual(Set(input.nodes.map { $0.address.localID }), Set(["observations", "cast", "split", "fit", "training", "validation"]))
        let fitted = try XCTUnwrap(input.nodes.first { $0.address.localID == "fit" })
        XCTAssertEqual(fitted.ports, Set(["training", "validation", "test", "state"]), "Staged inspection retains only configured outputs for captured inputs.")
        XCTAssertFalse(input.nodes.contains { $0.name == "Current renamed source" })
        XCTAssertEqual(input.terminalStatus, .completed)
        XCTAssertFalse(input.directory.path.hasPrefix(outcome.directory.path))
        let savedManifest = outcome.directory.appendingPathComponent("preparation/artifacts/manifest.json")
        let original = try Data(contentsOf: savedManifest)
        try Data("changed".utf8).write(to: savedManifest)
        XCTAssertThrowsError(try WorkflowHistoryReader.stagePreparationInspection(reference, root: root))
        // Inspection of the immutable staged snapshot remains independent of later archive edits.
        let runtime = try XCTUnwrap(workspace.runtime)
        let finished = expectation(description: "Actual archived inspection completes")
        var lines: [String] = [], exitCode: Int32?
        let process = try WorkflowExecutionSession.launchWorker(runtime.executableURL,
            runtime.argumentPrefix + input.arguments, runtime.workingDirectoryURL,
            { lines.append($0) }, { code, _ in exitCode = code; finished.fulfill() })
        defer { process.cancel() }
        await fulfillment(of: [finished], timeout: 30)
        XCTAssertEqual(exitCode, 0)
        var stream = WorkflowPreparationInspectionStream(input: input)
        for line in lines { try stream.receive(line) }
        let snapshot = try stream.finish(exitCode: try XCTUnwrap(exitCode))
        XCTAssertEqual(snapshot.outputs.count, 6)
        XCTAssertEqual(snapshot.outputs.mapValues { Set($0.keys) }, Dictionary(uniqueKeysWithValues: input.nodes.map { ($0.address, $0.ports) }), "Inspection contains exactly every captured node and configured output.")
        XCTAssertEqual(snapshot.cacheReport, input.cacheReport)
        XCTAssertTrue(snapshot.outputs.keys.allSatisfy { $0.domain == .data })
        XCTAssertTrue(snapshot.outputs.values.contains { ports in
            ports.values.contains { value in
                guard case .object(let output) = value, case .object(let preview) = output["preview"],
                      case .array(let rows) = preview["rows"] else { return false }
                return !rows.isEmpty && rows.count <= 8
            }
        })
        var failedExit = WorkflowPreparationInspectionStream(input: input)
        for line in lines { try failedExit.receive(line) }
        XCTAssertThrowsError(try failedExit.finish(exitCode: 1))
        XCTAssertThrowsError(try failedExit.finish(exitCode: 0))
        var incomplete = WorkflowPreparationInspectionStream(input: input)
        for line in lines.dropLast() { try incomplete.receive(line) }
        XCTAssertThrowsError(try incomplete.finish(exitCode: 0))
        XCTAssertThrowsError(try stream.finish(exitCode: 1))
        XCTAssertThrowsError(try stream.receive(try XCTUnwrap(lines.last)))
        var duplicate = WorkflowPreparationInspectionStream(input: input)
        let preview = try XCTUnwrap(lines.first { $0.contains("dataflowPreview") })
        try duplicate.receive(try XCTUnwrap(lines.first))
        try duplicate.receive(preview)
        XCTAssertThrowsError(try duplicate.receive(preview))
        let completion = try XCTUnwrap(lines.last)
        let mutations: [(inout [String: Any]) -> Void] = [
            { $0["manifestSHA256"] = String(repeating: "0", count: 64) },
            { $0["graphDigest"] = String(repeating: "0", count: 64) },
            { $0["provenance"] = ["historical": true, "sourceRevalidated": true, "cacheReuseVerified": false] },
            { $0["cache"] = [:] },
            { var provenance = $0["provenance"] as! [String: Any]; provenance["numpyVersion"] = "different-version"; $0["provenance"] = provenance }
        ]
        for mutate in mutations {
            var event = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(completion.utf8)) as? [String: Any])
            mutate(&event)
            var rejected = WorkflowPreparationInspectionStream(input: input)
            for line in lines.dropLast() { try rejected.receive(line) }
            let changed = String(decoding: try JSONSerialization.data(withJSONObject: event), as: UTF8.self)
            XCTAssertThrowsError(try rejected.receive(changed))
            XCTAssertThrowsError(try rejected.receive(completion))
            XCTAssertThrowsError(try rejected.finish(exitCode: 0))
        }
        for replacement: [String: Any] in [[:], ["kind": "state"], ["kind": "table", "preview": [:], "lineage": [:]]] {
            var rejected = WorkflowPreparationInspectionStream(input: input)
            var event = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(preview.utf8)) as? [String: Any])
            var ports = try XCTUnwrap(event["outputs"] as? [String: Any])
            ports[try XCTUnwrap(ports.keys.first)] = replacement
            event["outputs"] = ports
            try rejected.receive(try XCTUnwrap(lines.first))
            XCTAssertThrowsError(try rejected.receive(String(decoding: JSONSerialization.data(withJSONObject: event), as: UTF8.self)))
        }
        try original.write(to: savedManifest)
        let receiptHistory = WorkflowHistoryStore(workspace: workspace, host: host)
        receiptHistory.select(reference)
        let oldReceiptLoaded = expectation(description: "Original completed receipt loaded")
        let receiptObservation = receiptHistory.$isLoading.filter { !$0 }.prefix(1).sink { _ in oldReceiptLoaded.fulfill() }
        await fulfillment(of: [oldReceiptLoaded], timeout: 10); receiptObservation.cancel()
        // Completed preparation is useful even when training was subsequently cancelled.
        let receiptURL = outcome.directory.appendingPathComponent("outcome.json")
        var receipt = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: receiptURL)) as? [String: Any])
        receipt["status"] = "cancelled"
        receipt.removeValue(forKey: "modelManifestSHA256")
        receipt.removeValue(forKey: "workerRunID")
        let artifacts = try XCTUnwrap(receipt["artifacts"] as? [[String: Any]])
        // An interrupted fixture cannot claim completed training progress either.
        receipt["artifacts"] = artifacts.filter {
            !(($0["path"] as? String)?.hasPrefix("training/") ?? false) && $0["path"] as? String != "diagnostics/training-progress.json"
        }
        try JSONSerialization.data(withJSONObject: receipt).write(to: receiptURL)
        let cancelled = try WorkflowHistoryReader.stagePreparationInspection(reference, root: root)
        defer { cancelled.remove() }
        XCTAssertEqual(cancelled.terminalStatus, .cancelled)
        XCTAssertTrue(receiptHistory.inspectPreparation())
        let changedReceiptInspection = try XCTUnwrap(receiptHistory.preparationInspection)
        let changedReceiptDone = expectation(description: "Changed receipt inspection rejected")
        let changedObservation = changedReceiptInspection.$isBusy.filter { !$0 }.prefix(1).sink { _ in changedReceiptDone.fulfill() }
        await fulfillment(of: [changedReceiptDone], timeout: 30); changedObservation.cancel()
        XCTAssertNil(changedReceiptInspection.snapshot, "Inspection must not contradict the terminal status of selected History.")
        XCTAssertFalse(workspace.isBusy)
        receiptHistory.invalidate()
        let wholeStage = input.directory.deletingLastPathComponent().deletingLastPathComponent()
        input.remove()
        XCTAssertFalse(FileManager.default.fileExists(atPath: wholeStage.path))
        XCTAssertNil(input.storageReservation)
        XCTAssertNil(snapshot.input.storageReservation, "Copied display metadata shares released ownership.")
    }
}
