import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

final class WorkflowTrainedExportTests: XCTestCase {
    func testPackageFolderNameCannotEscapeSelectedDestination() throws {
        let parent = URL(fileURLWithPath: "/tmp/export-parent", isDirectory: true)
        for name in ["", " ", ".", "..", "../elsewhere", "/absolute", "nested/folder", "hidden\0name", String(repeating: "x", count: 129)] {
            XCTAssertThrowsError(try WorkflowTrainedExportPackage.destination(in: parent, folderName: name), name)
        }
        XCTAssertEqual(try WorkflowTrainedExportPackage.destination(in: parent, folderName: "My model").path,
                       "/tmp/export-parent/My model")
    }

    func testCancelledPublicationDoesNotCreateDestinationOrTemporarySibling() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: root) }
        let package = WorkflowTrainedExportPackage(reference: .init(projectID: UUID().uuidString, directoryID: UUID()),
            runID: UUID().uuidString, requestIdentity: "unused", modelManifestSHA256: "unused",
            directory: root, files: [], manifestBytes: Data(),
            inputContract: try TrainedModelInputContract(graph:ExampleFlows.parallelClassifier,registry:.builtIn(),columns:["x","y"],target:"label"))
        let operation = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            do { try package.publish(to: root.appendingPathComponent("export")); return false }
            catch is CancellationError { return true }
            catch { return false }
        }
        let cancelled = await operation.value
        XCTAssertTrue(cancelled)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), [])
    }

    @MainActor func testActualTrainedPackageIsPortableAndDoesNotCopySourceRows() async throws {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else { throw XCTSkip("Enable actual workflow runtime") }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: URL(fileURLWithPath: development), storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadPreparedWorkflowExample(device: "cpu"))
        let host = try XCTUnwrap(workspace.workflowSession)
        let done = expectation(description: "Training completes")
        let observation = host.$outcomes.dropFirst().filter { !$0.isEmpty }.prefix(1).sink { _ in done.fulfill() }
        XCTAssertTrue(host.run(train: WorkflowExamples.trainAddress))
        await fulfillment(of: [done], timeout: 60); observation.cancel()
        let outcome = try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(outcome.status, .completed)
        let reference = WorkflowHistoryReference(projectID: host.owner.projectID, directoryID: try XCTUnwrap(UUID(uuidString: outcome.directory.lastPathComponent)))
        let history = WorkflowHistoryStore(workspace: workspace, host: host)
        XCTAssertFalse(history.prepareTrainedExport(), "No selected verified model yet.")
        history.select(reference)
        let loaded = expectation(description: "History selected")
        let loading = history.$isLoading.filter { !$0 }.prefix(1).sink { _ in loaded.fulfill() }
        await fulfillment(of: [loaded], timeout: 10); loading.cancel()
        XCTAssertTrue(workspace.setExecutionPreferences(try .init(managedStorageLimitBytes: 1)))
        XCTAssertTrue(history.prepareTrainedExport())
        let refusedReview = try XCTUnwrap(history.trainedExport)
        let deadline = Date().addingTimeInterval(10)
        while refusedReview.isBusy && Date() < deadline { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertNil(refusedReview.package)
        XCTAssertTrue(refusedReview.statusMessage.lowercased().contains("storage"), refusedReview.statusMessage)
        XCTAssertTrue(workspace.resetExecutionPreferences())
        XCTAssertTrue(history.prepareTrainedExport())
        XCTAssertTrue(workspace.isBusy)
        let review = try XCTUnwrap(history.trainedExport)
        let ready = expectation(description: "Export prepared")
        let preparing = review.$isBusy.filter { !$0 }.prefix(1).sink { _ in ready.fulfill() }
        await fulfillment(of: [ready], timeout: 10); preparing.cancel()
        XCTAssertFalse(workspace.isBusy)
        let reviewDirectory = try XCTUnwrap(review.package?.directory)
        XCTAssertEqual(review.package?.runID, outcome.runID.uuidString)
        history.select(reference)
        XCTAssertNil(history.trainedExport)
        XCTAssertNil(review.package)
        XCTAssertFalse(FileManager.default.fileExists(atPath: reviewDirectory.path))
        XCTAssertFalse(review.save(to: root.appendingPathComponent("stale-export")))
        history.invalidate()
        let ledger = try WorkflowStorageAdmission(workspace: root)
        let baseline = try ledger.snapshot().committedBytes
        let measured = try WorkflowHistoryReader.stageTrainedExport(reference, root: root)
        let packageBytes = Int64(measured.files.reduce(measured.manifestBytes.count) { $0 + $1.bytes })
        measured.remove()
        XCTAssertEqual(try ledger.snapshot().committedBytes, baseline)
        let staging = root.appendingPathComponent("trained-export-staging")
        let beforeRefusal = try FileManager.default.contentsOfDirectory(atPath: staging.path)
        XCTAssertThrowsError(try WorkflowHistoryReader.stageTrainedExport(reference, root: root,
            storageLimitBytes: baseline + packageBytes - 1))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: staging.path), beforeRefusal)
        let writer = WorkflowTrainedExportWriter(storageLimitBytes: baseline + packageBytes)
        let package = try await writer.stage(reference, root: root)
        defer { package.remove() }
        XCTAssertTrue(package.directory.path.contains("/trained-export-staging/"))
        XCTAssertEqual(try ledger.snapshot().committedBytes, baseline + packageBytes)
        let scope = try WorkflowStorageAdmission.Scope(namespace: .trainedExportStaging, components: [package.directory.lastPathComponent])
        XCTAssertThrowsError(try ledger.reserve(scopes: [scope], peakBytes: 1, limitBytes: Int64.max),
            "Review must retain the exact reservation even when all admitted bytes have been written")
        XCTAssertEqual(Set(package.files.map(\.path)), ["model.py", "checkpoint.pt", "input-contract.json", "inference.py", "README.md", "requirements.txt"])
        XCTAssertEqual(package.runID, outcome.runID.uuidString)
        let contract = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: package.directory.appendingPathComponent("input-contract.json")))
        guard case .object(let input) = contract["input"] else { return XCTFail("Missing input contract") }
        XCTAssertEqual(input["featureColumns"], .array([.string("signal"), .string("context")]))
        XCTAssertEqual(input["preprocessing"], .string("alreadyPrepared"))
        XCTAssertEqual(package.inputContract.input,input)
        for name in package.files.map(\.path).filter({ $0 != "checkpoint.pt" }) + ["export-manifest.json"] {
            let text = try String(contentsOf: package.directory.appendingPathComponent(name), encoding: .utf8)
            XCTAssertFalse(text.contains(root.path)); XCTAssertFalse(text.contains(development))
            XCTAssertFalse(text.contains("trainingRowIDs")); XCTAssertFalse(text.contains("observationIDs"))
        }
        let checkpointRuntime = try XCTUnwrap(workspace.runtime)
        let privacyChecked = expectation(description: "Exported checkpoint excludes source records")
        var privacyCode: Int32?
        let privacyScript = """
        import sys, torch
        value = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
        assert set(value) == {'checkpointSchemaVersion','contract','modelState','optimizerState','epoch','seed','rngState'}
        forbidden = {'trainingRowIDs','validationRowIDs','observationIDs','sourcePath','sourceBindings','datasetCSV'}
        def inspect(item):
            if isinstance(item, str):
                assert all(private not in item for private in sys.argv[2:])
            elif isinstance(item, dict):
                assert not (set(item) & forbidden)
                for key, child in item.items(): inspect(key); inspect(child)
            elif isinstance(item, (list, tuple)):
                for child in item: inspect(child)
        inspect(value)
        """
        let privacyProcess = try WorkflowExecutionSession.launchWorker(checkpointRuntime.executableURL,
            ["-c", privacyScript, package.directory.appendingPathComponent("checkpoint.pt").path, root.path, development], root,
            { _ in }, { code, _ in privacyCode = code; privacyChecked.fulfill() })
        defer { privacyProcess.cancel() }
        await fulfillment(of: [privacyChecked], timeout: 20)
        XCTAssertEqual(privacyCode, 0)
        let checkpoint = outcome.directory.appendingPathComponent("training/artifacts/checkpoint.pt")
        let original = try Data(contentsOf: checkpoint)
        try Data("broken".utf8).write(to: checkpoint)
        XCTAssertThrowsError(try WorkflowHistoryReader.stageTrainedExport(reference, root: root))
        try original.write(to: checkpoint)
        let destination = root.appendingPathComponent("portable")
        try package.publish(to: destination)
        XCTAssertThrowsError(try package.publish(to: destination), "Existing exports must never be overwritten.")
        XCTAssertEqual(try Data(contentsOf: destination.appendingPathComponent("checkpoint.pt")), original)
        let stagedCode = package.directory.appendingPathComponent("model.py")
        try Data("changed staged code".utf8).write(to: stagedCode)
        let refused = root.appendingPathComponent("refused-export")
        XCTAssertThrowsError(try package.publish(to: refused))
        XCTAssertFalse(FileManager.default.fileExists(atPath: refused.path), "No partially successful export directory may remain.")
        XCTAssertEqual(try Data(contentsOf: destination.appendingPathComponent("checkpoint.pt")), original)
        package.remove()
        XCTAssertFalse(FileManager.default.fileExists(atPath: package.directory.path))
        XCTAssertEqual(try ledger.snapshot().committedBytes, baseline)
        try FileManager.default.removeItem(at: outcome.directory)
        let runtime = try XCTUnwrap(workspace.runtime)
        let predicted = expectation(description: "Standalone exported helper runs without original archive")
        var lines: [String] = [], code: Int32?
        let process = try WorkflowExecutionSession.launchWorker(runtime.executableURL,
            [destination.appendingPathComponent("inference.py").path, "--example"], root,
            { lines.append($0) }, { value, _ in code = value; predicted.fulfill() })
        defer { process.cancel() }
        await fulfillment(of: [predicted], timeout: 30)
        XCTAssertEqual(code, 0)
        let result = try JSONDecoder().decode([String: JSONValue].self, from: Data(try XCTUnwrap(lines.last).utf8))
        XCTAssertEqual(result["inputKind"], .string("synthetic prepared example"))
        guard case .array(let logits) = result["logits"] else { return XCTFail("Missing actual inference output") }
        XCTAssertEqual(logits.count, 2)
    }
}
