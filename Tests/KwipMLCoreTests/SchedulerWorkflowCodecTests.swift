import XCTest
@testable import KwipMLCore

final class SchedulerWorkflowCodecTests: XCTestCase {
    private func bytes(version: Int = 7, extra: Bool = false) throws -> Data {
        let payload: [String: Any] = ["kind": "schedulerRecipe", "type": "scheduler_step_lr", "configurationVersion": 1, "config": ["step_size": 2, "gamma": 0.5]]
        var root: [String: Any] = ["schemaVersion": version, "kind": "workflowProject", "projectID": UUID().uuidString, "revision": 0, "name": "Scheduler controls", "nodes": [["address": ["domain": "control", "localID": "schedule"], "name": "Smaller steps", "x": 0, "y": 0, "payload": payload]], "edges": [], "definitions": [], "drafts": []]
        if extra { root["schedule"] = "foreign" }
        return try JSONSerialization.data(withJSONObject: root)
    }
    func testNewSchedulerSourceRoundTripsWithExplicitFeatureVersion() throws {
        let document = try WorkflowCodec.decode(bytes())
        XCTAssertEqual(document.schemaVersion, 7)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)), document)
    }
    func testHistoricalSchemasAndForeignMetadataRefuseSchedulerFeatures() throws {
        for version in 1...6 { XCTAssertThrowsError(try WorkflowCodec.decode(bytes(version: version))) }
        XCTAssertThrowsError(try WorkflowCodec.decode(bytes(extra: true)))
        let old = WorkflowDocument(projectID: UUID().uuidString)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(old)).schemaVersion, 1)
    }
}
