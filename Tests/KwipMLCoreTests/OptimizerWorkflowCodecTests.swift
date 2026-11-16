import XCTest
@testable import KwipMLCore

final class OptimizerWorkflowCodecTests: XCTestCase {
    private func bytes(version: Int = 6, extra: Bool = false) throws -> Data {
        let payload: [String: Any] = ["kind": "optimizerRecipe", "type": "optimizer_sgd", "configurationVersion": 1, "config": ["lr": 0.1, "momentum": 0.8]]
        var root: [String: Any] = ["schemaVersion": version, "kind": "workflowProject", "projectID": UUID().uuidString, "revision": 0, "name": "Optimizer controls", "nodes": [["address": ["domain": "control", "localID": "recipe"], "name": "Momentum", "x": 0, "y": 0, "payload": payload]], "edges": [], "definitions": [], "drafts": []]
        if extra { root["optimizer"] = "foreign" }
        return try JSONSerialization.data(withJSONObject: root)
    }
    func testNewRecipeSourceRoundTripsWithoutChangingOldFiles() throws {
        let doc = try WorkflowCodec.decode(bytes())
        XCTAssertEqual(doc.schemaVersion, 6)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(doc)), doc)
        let old = WorkflowDocument(projectID: UUID().uuidString)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(old)).schemaVersion, 1)
    }
    func testOldSchemasAndForeignMetadataRefuseNewFeatures() throws {
        for version in 1...5 { XCTAssertThrowsError(try WorkflowCodec.decode(bytes(version: version))) }
        XCTAssertThrowsError(try WorkflowCodec.decode(bytes(extra: true)))
    }
}
