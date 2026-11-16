import XCTest
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class WorkflowWorkerLaunchBudgetTests: XCTestCase {
    private func root() throws -> URL {
        guard let path = realpath(FileManager.default.temporaryDirectory.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        defer { free(path) }
        let root = URL(fileURLWithPath: String(cString: path)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
    private func policy() throws -> ResolvedExecutionBudget { try .resolve(global: .factory, projectMaximumRunSeconds: 60) }
    private func evaluation(budget: Bool) throws -> WorkflowEvaluationWorkerRequest {
        .init(schemaVersion: budget ? 2 : 1, evaluationID: UUID().uuidString, projectID: UUID().uuidString,
              requestIdentity: String(repeating: "a", count: 64), snapshotReceiptSHA256: String(repeating: "b", count: 64),
              settings: .init(), executionBudget: try budget ? policy() : nil)
    }
    private func export(root: URL, budget: Bool) throws -> WorkflowEvaluationExportRequest {
        let project = UUID().uuidString, evaluation = UUID().uuidString
        let source = root.appendingPathComponent("workflow-evaluations/\(project)/\(evaluation)")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        return .init(schemaVersion: budget ? 2 : 1, exportID: UUID().uuidString, projectID: project, evaluationID: evaluation,
            requestIdentity: String(repeating: "a", count: 64), evaluationDirectory: source.path,
            evaluationManifestSHA256: String(repeating: "b", count: 64), includeTestRows: false,
            executionBudget: try budget ? policy() : nil)
    }
    func testWorkerRequestsStrictlyVersionPolicyWithoutAllowance() throws {
        let root = try root()
        for budget in [false, true] {
            let eval = try evaluation(budget: budget), export = try export(root: root, budget: budget)
            XCTAssertEqual(try WorkflowEvaluationWorkerRequest.decode(eval.encoded()), eval)
            XCTAssertEqual(try WorkflowEvaluationExportRequest.decode(export.encoded()), export)
            for (data, decode) in [(try eval.encoded(), { _ = try WorkflowEvaluationWorkerRequest.decode($0) }),
                                   (try export.encoded(), { _ = try WorkflowEvaluationExportRequest.decode($0) })] as [(Data, (Data) throws -> Void)] {
                let original = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
                XCTAssertEqual(original["executionBudget"] != nil, budget)
                var object = original; object["remainingSeconds"] = 30
                XCTAssertThrowsError(try decode(JSONSerialization.data(withJSONObject: object)))
                object = original; object["schemaVersion"] = budget ? 1 : 2
                XCTAssertThrowsError(try decode(JSONSerialization.data(withJSONObject: object)))
                object = original; object["executionBudget"] = NSNull()
                XCTAssertThrowsError(try decode(JSONSerialization.data(withJSONObject: object)))
                if budget {
                    object = original
                    var policy = object["executionBudget"] as! [String: Any]
                    policy["capabilityMaximumSeconds"] = NSNull(); object["executionBudget"] = policy
                    XCTAssertThrowsError(try decode(JSONSerialization.data(withJSONObject: object)))
                }
            }
        }
    }
    func testEvaluationRequiresImmutablePostStagingBudgetAndRetainsActivityOnFailure() throws {
        let request = try evaluation(budget: true)
        let owner = try WorkflowEvaluationOwnership.create(workspace: root(), request: request)
        defer { owner.close() }
        try request.encoded().write(to: owner.directory.appendingPathComponent("request.json"))
        XCTAssertThrowsError(try owner.prepareLaunch())
        for value in [0, -1, 61, Double.infinity, Double.nan] { XCTAssertThrowsError(try owner.publishLaunchBudget(remainingSeconds: value)) }
        let pin = try owner.publishLaunchBudget(remainingSeconds: 17.5)
        let raw = try Data(contentsOf: owner.directory.appendingPathComponent("launch-budget.json"))
        XCTAssertEqual(pin, WorkflowArchiveCodec.hash(raw))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: raw) as? [String: Any])
        XCTAssertEqual(Set(object.keys), ["schemaVersion", "requestSHA256", "remainingSeconds"])
        XCTAssertEqual(object["requestSHA256"] as? String, try request.requestSHA256())
        XCTAssertEqual(object["remainingSeconds"] as? Double, 17.5)
        XCTAssertThrowsError(try owner.publishLaunchBudget(remainingSeconds: 10))
        try Data("{}".utf8).write(to: owner.directory.appendingPathComponent("launch-budget.json"))
        XCTAssertThrowsError(try owner.prepareLaunch())
        XCTAssertThrowsError(try WorkflowFileLease(directory: owner.directory, name: "activity.lock", mode: .shared))
        try raw.write(to: owner.directory.appendingPathComponent("launch-budget.json"))
        XCTAssertEqual(try owner.prepareLaunch(), owner.ownershipSHA256)
        XCTAssertThrowsError(try owner.publishLaunchBudget(remainingSeconds: 5))
    }
    func testExportBudgetPinsRequestAndCannotBeForged() throws {
        let root = try root(), request = try export(root: root, budget: true)
        let owner = try WorkflowEvaluationExportOwnership.create(workspace: root, request: request)
        defer { owner.close() }
        try owner.publishRequest(request)
        XCTAssertThrowsError(try owner.prepareLaunch())
        let pin = try owner.publishLaunchBudget(remainingSeconds: 20)
        let path = owner.directory.appendingPathComponent("launch-budget.json"), raw = try Data(contentsOf: path)
        XCTAssertEqual(pin, WorkflowArchiveCodec.hash(raw))
        let original = try XCTUnwrap(JSONSerialization.jsonObject(with: raw) as? [String: Any])
        for (key, value) in [("requestSHA256", String(repeating: "f", count: 64)), ("remainingSeconds", 61), ("remainingSeconds", true), ("extra", 1)] as [(String, Any)] {
            var object = original; object[key] = value
            try JSONSerialization.data(withJSONObject: object).write(to: path)
            XCTAssertThrowsError(try owner.prepareLaunch())
        }
        try raw.write(to: path)
        XCTAssertEqual(try owner.prepareLaunch(), owner.ownershipSHA256)
    }
    func testLegacyOwnersRejectUnexpectedLaunchBudgetWithoutReleasingActivity() throws {
        let root = try root(), eval = try evaluation(budget: false), export = try export(root: root, budget: false)
        let evalOwner = try WorkflowEvaluationOwnership.create(workspace: root, request: eval)
        let exportOwner = try WorkflowEvaluationExportOwnership.create(workspace: root, request: export)
        defer { evalOwner.close(); exportOwner.close() }
        try eval.encoded().write(to: evalOwner.directory.appendingPathComponent("request.json"))
        try exportOwner.publishRequest(export)
        XCTAssertThrowsError(try evalOwner.publishLaunchBudget(remainingSeconds: 5))
        XCTAssertThrowsError(try exportOwner.publishLaunchBudget(remainingSeconds: 5))
        for directory in [evalOwner.directory, exportOwner.directory] {
            try Data("{}".utf8).write(to: directory.appendingPathComponent("launch-budget.json"))
        }
        XCTAssertThrowsError(try evalOwner.prepareLaunch())
        XCTAssertThrowsError(try exportOwner.prepareLaunch())
    }
}
