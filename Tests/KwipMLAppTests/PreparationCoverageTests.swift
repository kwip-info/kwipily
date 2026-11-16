import XCTest
import KwipMLCore
@testable import KwipMLApp

final class PreparationCoverageTests: XCTestCase {
    private func bytes(_ object: Any) throws -> Data { try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .withoutEscapingSlashes]) }
    private func fixture() throws -> (Data, Data, Data) {
        let graph = DataFlowDocument(nodes: [
            .init(id: "source", operatorID: "kwip.data.sourceCSV"),
            .init(id: "full", operatorID: "kwip.data.columns", config: ColumnsConfigurationDraft.defaults),
            .init(id: "sample", operatorID: "kwip.data.columns", config: ColumnsConfigurationDraft.defaults)], edges: [
                .init(from: .init(node: "source", port: "table"), to: .init(node: "full", port: "table")),
                .init(from: .init(node: "source", port: "table"), to: .init(node: "sample", port: "table"))])
        let graphBytes = try DataFlowCodec.encodeGraph(graph), pin = WorkflowArchiveCodec.hash(graphBytes)
        let demand: [String: Any] = ["schemaVersion": 1, "graphDigest": pin,
            "sampling": ["schemaVersion": 1, "mode": "rowCount", "rowCount": 3],
            "selection": "sourceOrderPrefix", "completeNodeIDs": ["source", "full"],
            "sampledNodeIDs": ["sample"], "percentageRequiresCardinalityScan": false]
        let plan: [String: Any] = ["schemaVersion": 2, "graphDigest": pin,
            "graph": try JSONSerialization.jsonObject(with: graphBytes), "demand": demand]
        let planBytes = try bytes(plan)
        var coverage = demand
        coverage["status"] = "sampled"
        coverage["nodes"] = ["source": ["completeness": "complete", "rowCounts": ["table": 40]],
                             "full": ["completeness": "complete", "rowCounts": ["table": 40]],
                             "sample": ["completeness": "sampled", "rowCounts": ["table": 3]]]
        let artifacts: [[String: Any]] = ["source", "full", "sample"].map { ["kind": "table", "nodeID": $0, "port": "table"] }
        let manifest: [String: Any] = ["schemaVersion": 1, "kind": "dataflowExecution", "status": "completed",
            "graphDigest": pin, "preparationPlanDigest": WorkflowArchiveCodec.hash(try WorkflowPreparationCoverage.canonicalUTF8(planBytes)),
            "executionCoverage": coverage, "artifacts": artifacts]
        return (try bytes(manifest), graphBytes, planBytes)
    }
    func testVersionedWorkloadIsPinnedByPlanAndAcceptedOperation() throws {
        let (manifest, graph, plan) = try fixture()
        let policy = try ExecutionWorkloadPolicy(preset: .balanced)
        let encoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(policy))
        var captured = try XCTUnwrap(JSONSerialization.jsonObject(with: plan) as? [String: Any])
        captured["schemaVersion"] = 3; captured["workloadPolicy"] = encoded
        let planBytes = try bytes(captured)
        var envelope = try XCTUnwrap(JSONSerialization.jsonObject(with: manifest) as? [String: Any])
        envelope["preparationPlanDigest"] = WorkflowArchiveCodec.hash(try WorkflowPreparationCoverage.canonicalUTF8(planBytes))
        envelope["workloadPolicy"] = encoded
        XCTAssertNoThrow(try WorkflowPreparationCoverage.verify(manifest: bytes(envelope), graph: graph, plan: planBytes, allowSample: true))
        XCTAssertNoThrow(try WorkflowCompletedPreparationInventory.verifyWorkload(data: bytes(envelope), expected: policy))
        XCTAssertThrowsError(try WorkflowCompletedPreparationInventory.verifyWorkload(data: bytes(envelope), expected: nil))
        XCTAssertThrowsError(try WorkflowCompletedPreparationInventory.verifyWorkload(data: bytes(envelope), expected: ExecutionWorkloadPolicy(preset: .light)))
        envelope.removeValue(forKey: "workloadPolicy")
        XCTAssertThrowsError(try WorkflowPreparationCoverage.verify(manifest: bytes(envelope), graph: graph, plan: planBytes, allowSample: true))
    }
    func testMixedSuccessfulPreparationKeepsCompleteBranchAndSampleDistinct() throws {
        let (manifest, graph, plan) = try fixture()
        let inspected = try XCTUnwrap(WorkflowPreparationCoverage.verify(manifest: manifest, graph: graph, plan: plan, allowSample: true))
        XCTAssertTrue(inspected.isSampled)
        XCTAssertTrue(inspected.isComplete("full"))
        XCTAssertFalse(inspected.isComplete("sample"))
        XCTAssertNoThrow(try WorkflowPreparationCoverage.verify(manifest: manifest, graph: graph, plan: plan, selectedNodes: ["full"]))
        XCTAssertThrowsError(try WorkflowPreparationCoverage.verify(manifest: manifest, graph: graph, plan: plan, selectedNodes: ["sample"]))
        XCTAssertThrowsError(try WorkflowPreparationCoverage.verify(manifest: manifest, graph: graph, plan: plan))
    }
    func testCompletedStatusCannotHideRemovedOrRelabelledCoverage() throws {
        let (manifest, graph, plan) = try fixture()
        var value = try JSONSerialization.jsonObject(with: manifest) as! [String: Any]
        value.removeValue(forKey: "executionCoverage")
        XCTAssertThrowsError(try WorkflowPreparationCoverage.verify(manifest: bytes(value), graph: graph, plan: plan, allowSample: true))
        value = try JSONSerialization.jsonObject(with: manifest) as! [String: Any]
        var coverage = value["executionCoverage"] as! [String: Any]
        coverage["status"] = "complete"; coverage["sampledNodeIDs"] = [String](); coverage["completeNodeIDs"] = ["source", "full", "sample"]
        value["executionCoverage"] = coverage
        XCTAssertThrowsError(try WorkflowPreparationCoverage.verify(manifest: bytes(value), graph: graph, plan: plan, allowSample: true))
    }
    func testArchivedPlanTamperingAndLegacyFullCompatibility() throws {
        let (manifest, graph, plan) = try fixture()
        var changed = try JSONSerialization.jsonObject(with: plan) as! [String: Any]
        changed["schemaVersion"] = 1
        XCTAssertThrowsError(try WorkflowPreparationCoverage.verify(manifest: manifest, graph: graph, plan: bytes(changed), allowSample: true))
        XCTAssertNil(try WorkflowPreparationCoverage.verify(manifest: bytes(["graphDigest": WorkflowArchiveCodec.hash(graph)]), graph: graph, plan: nil))
    }
    func testPythonCanonicalPlanPreservesNumericTokensAndUnicode() throws {
        let encoded = Data(#"{"nested":{"line":"x\n"},"number":1.0,"text":"caf\u00e9\ud83d\ude00"}"#.utf8)
        let expected = Data(#"{"nested":{"line":"x\n"},"number":1.0,"text":"café😀"}"#.utf8)
        XCTAssertEqual(try WorkflowPreparationCoverage.canonicalUTF8(encoded), expected)
    }
    func testOriginalCapturedSamplingCannotBeDowngradedOrReplaced() throws {
        let (manifest, graph, plan) = try fixture()
        let sampling: JSONValue = .object(["schemaVersion": .integer(1), "mode": .string("rowCount"), "rowCount": .integer(3)])
        XCTAssertNoThrow(try WorkflowPreparationCoverage.verify(manifest: manifest, graph: graph, plan: plan, allowSample: true,
            expectedSampling: sampling, expectedCompleteNodes: ["full"]))
        XCTAssertThrowsError(try WorkflowPreparationCoverage.verify(manifest: manifest, graph: graph, plan: plan, allowSample: true,
            expectedSampling: .object(["schemaVersion": .integer(1), "mode": .string("rowCount"), "rowCount": .integer(4)]), expectedCompleteNodes: ["full"]))
        XCTAssertThrowsError(try WorkflowPreparationCoverage.verify(manifest: manifest, graph: graph, plan: plan, allowSample: true,
            expectedSampling: sampling, expectedCompleteNodes: []))
        XCTAssertThrowsError(try WorkflowPreparationCoverage.verify(manifest: bytes(["graphDigest": WorkflowArchiveCodec.hash(graph)]), graph: graph, plan: nil,
            allowSample: true, expectedSampling: sampling, expectedCompleteNodes: []))
    }
}
