import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class BackendFitSessionTests: XCTestCase {
    func testRealMissingSourceColumnIsAttributedDuringEstimatorPreparation() async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_SKLEARN_TEST_PYTHON"] else {throw XCTSkip("Select the pinned sklearn interpreter")}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=try canonicalBackendTemporaryDirectory()
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        let document=try WorkflowTemplates.instantiate(id:"customer-segmentation")
        _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        workspace.sklearnRuntimeCapability=try .decode(JSONEncoder().encode(SklearnRuntimeCapabilities.expected))
        let source=root.appendingPathComponent("missing-customer.csv")
        try Data("InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,Country\n123,A,Test,1,2011-01-01,2,UK\n".utf8).write(to:source)
        let address=WorkflowNodeAddress(domain:.data,localID:"source")
        XCTAssertTrue(host.bindings.setPath(source.path,for:address,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        host.executionLaunch={_,arguments,directory,onLine,onExit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),arguments,directory,onLine,onExit)}
        XCTAssertTrue(host.run(train:.init(domain:.control,localID:"segments")),host.statusMessage)
        let deadline=Date().addingTimeInterval(60)
        while host.backendFit?.isRunning==true && Date()<deadline {try await Task.sleep(for:.milliseconds(30))}
        let session=try XCTUnwrap(host.backendFit)
        XCTAssertEqual(session.phase,.failed)
        XCTAssertTrue(session.status.contains("CustomerID"),session.status)
        let evidence=try XCTUnwrap(host.attributedPreparationFailure)
        XCTAssertEqual(evidence.detail.nodeID,"columns")
        XCTAssertTrue(evidence.isCurrent)
        let display=try XCTUnwrap(host.canvas.projection.displayIDByAddress[evidence.address])
        XCTAssertEqual(host.canvas.canvasNodeFeedback(for:display).executionFailure,evidence.detail.cause)
        XCTAssertNil(session.output);XCTAssertTrue(host.backendHistory.entries.isEmpty)
        let replacement=root.appendingPathComponent("replacement.csv");try FileManager.default.copyItem(at:source,to:replacement)
        XCTAssertTrue(host.bindings.setPath(replacement.path,for:address,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        XCTAssertFalse(try XCTUnwrap(host.attributedPreparationFailure).isCurrent)
        XCTAssertNil(host.canvas.canvasNodeFeedback(for:display).executionFailure)
    }
    func testRealCanvasPreparationFitHistoryAndOutput() async throws {
        guard let python = ProcessInfo.processInfo.environment["KWIPML_SKLEARN_TEST_PYTHON"] else { throw XCTSkip("Select an isolated pinned sklearn interpreter.") }
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = try canonicalBackendTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadBackendWorkflowExample(backendID: "sklearn"), workspace.statusMessage)
        let host = try XCTUnwrap(workspace.workflowSession)
        // Real worker execution below uses the isolated pinned sklearn environment;
        // package/runtime discovery has its own strict capability tests.
        workspace.sklearnRuntimeCapability = try .decode(JSONEncoder().encode(SklearnRuntimeCapabilities.expected))
        host.executionLaunch = { _, arguments, workingDirectory, onLine, onExit in
            try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath: python), arguments, workingDirectory, onLine, onExit)
        }
        XCTAssertTrue(host.run(train: WorkflowExamples.sklearnEstimatorAddress), host.statusMessage)
        let deadline = Date().addingTimeInterval(60)
        while host.backendFit?.isRunning == true && Date() < deadline { try await Task.sleep(nanoseconds: 100_000_000) }
        let session = try XCTUnwrap(host.backendFit)
        XCTAssertFalse(session.isRunning, session.status)
        XCTAssertEqual(session.phase, .completed, session.status + "\n" + session.messages.joined(separator: "\n"))
        XCTAssertNotNil(session.output)
        let estimatorDisplay = try XCTUnwrap(host.canvas.projection.displayIDByAddress[WorkflowExamples.sklearnEstimatorAddress])
        XCTAssertEqual(host.canvas.previewStatus(for: estimatorDisplay), "Completed")
        XCTAssertFalse(host.owner.isBusy)
        XCTAssertFalse(workspace.isBusy)
        XCTAssertNotNil(host.snapshot(for: .init(domain: .data, localID: "fit")))
        let training=WorkflowNodeAddress(domain:.data,localID:"fit")
        XCTAssertTrue(host.owner.select(.init(nodeAddresses:[training]),lease:host.owner.lease()))
        let viewer=try XCTUnwrap(host.capturedTableViewSession(for:.init(address:training,port:"training")))
        XCTAssertTrue(viewer.load())
        let queryDeadline=Date().addingTimeInterval(20)
        while viewer.isBusy && Date()<queryDeadline {try await Task.sleep(for:.milliseconds(20))}
        XCTAssertNotNil(viewer.page,viewer.statusMessage)
        XCTAssertGreaterThan(viewer.page?.totalRows ?? 0,8,"Backend-fit table browsing must exceed its transport sample")
        XCTAssertEqual(host.backendHistory.entries.count, 1, host.backendHistory.status)
        if let entry = host.backendHistory.entries.first {
            let verified = try await host.backendHistory.verify(entry)
            XCTAssertTrue(FileManager.default.fileExists(atPath: verified.artifactDirectory.appendingPathComponent("portable/inference.py").path))
            let restored = BackendFitHistoryStore(root: root, projectID: host.owner.projectID)
            await restored.reloadAndWait()
            XCTAssertEqual(restored.entries.map(\.id), [entry.id], restored.status)
        }
        XCTAssertTrue(host.owner.rename("Changed after fitting", lease: host.owner.lease()))
        XCTAssertEqual(host.canvas.previewStatus(for: estimatorDisplay), "Completed","Labels preserve captured results")
        let partition=WorkflowNodeAddress(domain:.data,localID:"split")
        guard case .dataOperator(_,var config)=host.owner.document.nodes.first(where:{$0.address==partition})?.payload else {return XCTFail("Missing Partition fixture")}
        config["seed"] = .integer(99)
        XCTAssertTrue(host.canvas.apply(.configure(address:partition,config:config),lease:host.owner.lease()))
        XCTAssertEqual(host.canvas.previewStatus(for: estimatorDisplay), "Stale")
        XCTAssertNil(host.capturedTableViewSession(for:.init(address:training,port:"training")))
        XCTAssertFalse(viewer.load())

    }
}
