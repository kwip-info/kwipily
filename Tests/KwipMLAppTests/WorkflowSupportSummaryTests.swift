import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowSupportSummaryTests: XCTestCase {
    func testPrivateFieldsNeverEnterSummaryAndPolicyIsOptional() throws {
        let secret = "PRIVATE-CANARY-/Users/person/dataset.csv"
        let outcome = WorkflowArchiveOutcome(projectID: secret, directoryID: secret, runID: secret,
            requestIdentity: secret, status: .failed, terminalPhase: .training,
            startedAt: secret, finishedAt: secret, artifacts: [.init(path: secret, bytes: 17, sha256: secret)],
            workerRunID: secret, error: secret, failureCode: .resourceBudgetFailed)
        let policy = try ExecutionPreferences(cpuThreads: 3, managedStorageLimitBytes: 2000).resolve().budget
        for include in [false, true] {
            let bytes = try WorkflowSupportSummary(outcome: outcome, budget: policy, sampleCapture: .factory,
                includeExecutionPolicy: include).encoded()
            let text = String(decoding: bytes, as: UTF8.self)
            XCTAssertFalse(text.contains(secret)); XCTAssertFalse(text.contains("/Users"))
            let object = try XCTUnwrap(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
            XCTAssertEqual(Set(object.keys), Set(["schemaVersion","operation","status","terminalPhase","artifactCount","failureCode"]).union(include ? ["executionBudget","sampleCapture"] : []))
            XCTAssertEqual(object["failureCode"] as? String, "resourceBudgetFailed")
            XCTAssertEqual(object["artifactCount"] as? Int, 1)
            XCTAssertLessThan(bytes.count, 16384)
        }
    }
    func testLegacyAbsentPolicyStaysUnknown() throws {
        let outcome = WorkflowArchiveOutcome(projectID: "p", directoryID: "d", runID: "r", requestIdentity: "i",
            status: .completed, terminalPhase: .training, startedAt: "", finishedAt: "", artifacts: [])
        let bytes = try WorkflowSupportSummary(outcome: outcome, budget: nil, sampleCapture: nil).encoded()
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
        XCTAssertNil(object["executionBudget"]); XCTAssertNil(object["sampleCapture"]); XCTAssertNil(object["failureCode"])
    }
}
