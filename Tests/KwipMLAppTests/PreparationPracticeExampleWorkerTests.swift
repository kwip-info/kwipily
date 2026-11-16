import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

final class PreparationPracticeExampleWorkerTests: XCTestCase {
    @MainActor func testActualPreparationReusesTrainingStatisticsAndIgnoresUnknownCategory() async throws {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {
            throw XCTSkip("Set KWIPML_TEST_DEVELOPMENT_ROOT for the real preparation practice workflow.")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).standardizedFileURL.resolvingSymlinksInPath()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let workspace = WorkspaceStore(rootOverride: URL(fileURLWithPath: development), storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        let document = try PreparedWorkflowExample.preparationPractice.document(device: "cpu")
        _ = try workspace.projectLibrary.saveSource(.workflow(document), id: document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id: document.projectID), workspace.statusMessage)
        let host = try XCTUnwrap(workspace.workflowSession)
        let csv = root.appendingPathComponent("observations.csv")
        try FileManager.default.copyItem(at: PreparationPracticeExample.observationsURL(), to: csv)
        XCTAssertTrue(host.bindings.setPath(csv.path, for: WorkflowExamples.sourceAddress,
            lease: host.owner.lease(), expectedRevision: host.bindings.revision))
        let completed = expectation(description: "Actual practice training terminates")
        let observation = host.$outcomes.filter { !$0.isEmpty }.prefix(1).sink { _ in completed.fulfill() }
        defer { observation.cancel() }
        XCTAssertTrue(host.run(train: WorkflowExamples.trainAddress), host.statusMessage)
        await fulfillment(of: [completed], timeout: 60)
        if workspace.isBusy { host.cancel() }
        let outcome = try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(outcome.status, .completed, outcome.error ?? "")
        XCTAssertNil(outcome.recordError)
        XCTAssertNotNil(outcome.modelManifestSHA256)
        let artifacts = outcome.directory.appendingPathComponent("preparation/artifacts")
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: artifacts.appendingPathComponent("manifest.json"))) as? [String: Any])
        let entries = try XCTUnwrap(manifest["artifacts"] as? [[String: Any]])
        func artifact(_ node: String, _ port: String) throws -> [String: Any] {
            let entry = try XCTUnwrap(entries.first { $0["nodeID"] as? String == node && $0["port"] as? String == port })
            let path = try XCTUnwrap(entry["path"] as? String)
            return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: artifacts.appendingPathComponent(path))) as? [String: Any])
        }
        let state=try artifact("prepare","state")
        let rules=try XCTUnwrap(state["rules"] as? [[String:Any]])
        let amountRule=try XCTUnwrap(rules.first {$0["column"] as? String == "amount"})
        // Independent arithmetic: (2+4+8+10+14+16)/6=9; held-out extremes are excluded.
        XCTAssertEqual((amountRule["fill"] as? NSNumber)?.doubleValue,9)
        let regionRule=try XCTUnwrap(rules.first {$0["column"] as? String == "region"})
        XCTAssertEqual(regionRule["vocabulary"] as? [String],["north","south"])
        let training=try artifact("prepare","training")
        let heldout=try artifact("prepare","validation")
        let test=try artifact("prepare","test")
        XCTAssertEqual((test["rows"] as? [[Any]])?.count,4)
        let trainingRows = try XCTUnwrap(training["rows"] as? [[Any]])
        let heldoutRows = try XCTUnwrap(heldout["rows"] as? [[Any]])
        XCTAssertEqual(trainingRows.count, 8)
        XCTAssertEqual(heldoutRows.count, 4)
        let columns = try XCTUnwrap(heldout["columns"] as? [[String: Any]])
        let amount = try XCTUnwrap(columns.firstIndex { $0["name"] as? String == "amount" })
        let north = try XCTUnwrap(columns.firstIndex { $0["name"] as? String == "onehot:[\"region\",\"north\"]" })
        let south = try XCTUnwrap(columns.firstIndex { $0["name"] as? String == "onehot:[\"region\",\"south\"]" })
        XCTAssertEqual(trainingRows.map { ($0[amount] as! NSNumber).doubleValue }, [2,4,9,8,10,9,14,16])
        XCTAssertEqual(heldoutRows.map { ($0[amount] as! NSNumber).doubleValue }, [1000,9,2000,3000])
        XCTAssertEqual(heldoutRows.map { ($0[north] as! NSNumber).doubleValue }, [0,0,0,0])
        XCTAssertEqual(heldoutRows.map { ($0[south] as! NSNumber).doubleValue }, [0,0,1,0])
        let trainLineage = try XCTUnwrap(training["lineage"] as? [String: Any])
        let validationLineage = try XCTUnwrap(heldout["lineage"] as? [String: Any])
        XCTAssertEqual(trainLineage["fittedStateDigests"] as? [String], validationLineage["fittedStateDigests"] as? [String])
        XCTAssertEqual((validationLineage["fittedStateDigests"] as? [String])?.count, 1)
        let imputation=try XCTUnwrap(trainLineage["diagnostics"] as? [[String:Any]])
        XCTAssertTrue(imputation.contains {$0["code"] as? String == "missingValuesImputed" && $0["column"] as? String == "amount" && $0["count"] as? Int == 2})
        let diagnostics = try XCTUnwrap(validationLineage["diagnostics"] as? [[String: Any]])
        XCTAssertTrue(diagnostics.contains { $0["code"] as? String == "unknownCategoryIgnored" && $0["count"] as? Int == 3 })
        let reference = WorkflowHistoryReference(projectID: document.projectID, directoryID: try XCTUnwrap(UUID(uuidString: outcome.directory.lastPathComponent)))
        guard case .verified = WorkflowHistoryReader.load(reference, root: root) else { return XCTFail("Completed run must retain verified History") }
        workspace.refreshDraftCode()
        let codeDeadline = Date().addingTimeInterval(30)
        while workspace.isGeneratingCode && Date() < codeDeadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertFalse(workspace.isGeneratingCode, "Code export must terminate")
        XCTAssertNil(workspace.draftCodeError)
        XCTAssertFalse(workspace.draftCode.isEmpty, "The prepared model must export usable source")
        // Entire-workflow Code is the default: the model is captured alongside
        // preparation and Train settings, and the driver invokes the shared runtime.
        XCTAssertEqual(workspace.workflowCodeScope, .workflow)
        XCTAssertEqual(workspace.draftCodeFiles["workflow.py"], workspace.draftCode)
        XCTAssertTrue(workspace.draftCode.contains("def main():"))
        XCTAssertTrue(workspace.draftCode.contains("execute_snapshot(SNAPSHOT, bindings, args.output)"))
        let sourcePlan = try JSONDecoder().decode(WorkflowCodePlan.self, from: Data(try XCTUnwrap(workspace.draftCodeFiles["workflow-plan.json"]).utf8))
        try sourcePlan.validate()
        let expectedPlan = try WorkflowCodePlan.compile(host.owner.document, scope: .workflow,
            modelRegistry: host.owner.modelRegistry, dataRegistry: host.owner.dataRegistry,
            sampling: workspace.preparationSampling)
        XCTAssertEqual(sourcePlan, expectedPlan, "Code must capture the complete current workflow, including exact model, preparation, Train settings and selected recipes")
        XCTAssertEqual(sourcePlan.trainings.map(\.id), [WorkflowExamples.trainAddress.localID])
        XCTAssertNotNil(workspace.draftCodeFiles["preparation.py"])
        let prepareAddress=WorkflowNodeAddress(domain:.data,localID:"prepare")
        let prepareNode=try XCTUnwrap(host.owner.document.nodes.first {$0.address == prepareAddress})
        guard case .dataOperator(let operatorID,let configuration)=prepareNode.payload else {return XCTFail("Missing Preprocess")}
        XCTAssertEqual(operatorID,"kwip.data.preprocess")
        var strict=try PreprocessConfiguration(values:configuration)
        let region=try XCTUnwrap(strict.columns.firstIndex {$0.column == "region"})
        strict.columns[region].unknownPolicy="error"
        XCTAssertNotNil(host.owner.apply(.configure(address:prepareAddress,config:strict.values),lease:host.owner.lease()))
        let previousCount = host.outcomes.count
        let rejected = expectation(description: "Strict unknown-category policy rejects held-out rows")
        let rejectionObservation = host.$outcomes.filter { $0.count > previousCount }.prefix(1).sink { _ in rejected.fulfill() }
        defer { rejectionObservation.cancel() }
        XCTAssertTrue(host.run(train: WorkflowExamples.trainAddress), host.statusMessage)
        await fulfillment(of: [rejected], timeout: 60)
        if workspace.isBusy { host.cancel() }
        let failure = try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(failure.status, .failed)
        XCTAssertTrue((failure.error ?? "").contains("Unknown category"), failure.error ?? "")
        XCTAssertNil(failure.modelManifestSHA256)
        guard case .verified = WorkflowHistoryReader.load(reference, root: root) else { return XCTFail("Failure must not invalidate the previous completed run") }

    }
}
