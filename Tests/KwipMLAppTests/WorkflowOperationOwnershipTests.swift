import XCTest
import KwipMLCore
import Darwin
@testable import KwipMLApp

final class WorkflowOperationOwnershipTests: XCTestCase {
    private func fixture(budget: Bool = false) throws -> (URL, WorkflowOperationRequest) {
        guard let canonical = realpath(FileManager.default.temporaryDirectory.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        defer { free(canonical) }
        let root = URL(fileURLWithPath: String(cString: canonical)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let project = UUID().uuidString, directory = UUID().uuidString
        let phase = root.appendingPathComponent("workflow-runs/\(project)/\(directory)/preparation")
        let policy = try budget ? ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: 60) : nil
        return (root, WorkflowOperationRequest(operationID: UUID().uuidString, projectID: project,
            runID: UUID().uuidString, directoryID: directory, requestIdentity: String(repeating: "a", count: 64),
            phase: .preparation, arguments: ["--execute-dataflow", phase.appendingPathComponent("request.json").path,
                "--source-bindings", phase.appendingPathComponent("bindings.json").path,
                "--output", phase.appendingPathComponent("artifacts").path],
            executionBudget: policy, remainingSeconds: budget ? 12.5 : nil))
    }
    func testLegacyAndBudgetRequestsRoundTripAndEnforceWorkspace() throws {
        for budget in [false, true] {
            let (root, request) = try fixture(budget: budget)
            try request.validate(workspace: root)
            XCTAssertEqual(try WorkflowOperationRequest.decode(request.encoded()), request)
            XCTAssertThrowsError(try request.validate(workspace: root.appendingPathComponent("other")))
            let object = try XCTUnwrap(JSONSerialization.jsonObject(with: request.encoded()) as? [String: Any])
            XCTAssertEqual(object["executionBudget"] != nil, budget)
            XCTAssertEqual(object["remainingSeconds"] != nil, budget)
        }
    }
    func testRejectsMalformedPolicyPairsUnknownFieldsAndArguments() throws {
        let (root, request) = try fixture(budget: true)
        let original = try XCTUnwrap(JSONSerialization.jsonObject(with: request.encoded()) as? [String: Any])
        for change: (inout [String: Any]) -> Void in [
            { $0["remainingSeconds"] = 0 }, { $0["remainingSeconds"] = 61 },
            { $0["remainingSeconds"] = true },
            { var policy = $0["executionBudget"] as! [String: Any]; policy["capabilityMaximumSeconds"] = NSNull(); $0["executionBudget"] = policy }, { $0["remainingSeconds"] = NSNull() },
            { $0.removeValue(forKey: "remainingSeconds") }, { $0.removeValue(forKey: "executionBudget") },
            { $0["extra"] = true }, { $0["operationID"] = UUID().uuidString.lowercased() },
            { $0["phase"] = "evaluation" }, { $0["arguments"] = ["--arbitrary", "value"] },
            { $0["arguments"] = request.arguments + ["--output", "/tmp/escape"] }
        ] {
            var object = original; change(&object)
            XCTAssertThrowsError(try WorkflowOperationRequest.decode(JSONSerialization.data(withJSONObject: object)))
        }
        var wrong = request; wrong.arguments[1] = root.appendingPathComponent("elsewhere/request.json").path
        XCTAssertThrowsError(try wrong.validate(workspace: root))
    }
    func testOwnerPinsRequestAndRetainsExclusiveAuthorityUntilClose() throws {
        let (root, request) = try fixture(budget: true)
        let owner = try WorkflowOperationOwnership.create(workspace: root, request: request)
        defer { owner.close() }
        XCTAssertEqual(owner.directory.path, root.appendingPathComponent("workflow-operations/\(request.projectID)/\(request.operationID)").path)
        XCTAssertEqual(owner.metadata.kind, "workflowOperation")
        XCTAssertEqual(owner.metadata.requestSHA256, try request.requestSHA256())
        XCTAssertThrowsError(try WorkflowFileLease(directory: owner.directory, name: "owner.lock", mode: .shared))
        XCTAssertThrowsError(try WorkflowFileLease(directory: owner.directory, name: "activity.lock", mode: .shared))
        XCTAssertThrowsError(try owner.prepareLaunch())
        try owner.publishRequest(request)
        XCTAssertEqual(try owner.prepareLaunch(), owner.ownershipSHA256)
        let activity = try WorkflowFileLease(directory: owner.directory, name: "activity.lock", mode: .shared)
        activity.close()
        XCTAssertThrowsError(try WorkflowFileLease(directory: owner.directory, name: "owner.lock", mode: .shared))
        XCTAssertThrowsError(try owner.prepareLaunch())
        XCTAssertThrowsError(try owner.publishRequest(request))
        XCTAssertThrowsError(try WorkflowOperationOwnership.create(workspace: root, request: request))
        owner.close(); owner.close()
        XCTAssertThrowsError(try owner.check())
        let released = try WorkflowFileLease(directory: owner.directory, name: "owner.lock", mode: .exclusive)
        released.close()
    }
    func testChangedRequestOrExistingClaimNeverReleasesActivity() throws {
        let (root, request) = try fixture()
        let owner = try WorkflowOperationOwnership.create(workspace: root, request: request)
        defer { owner.close() }
        try owner.publishRequest(request)
        let path = owner.directory.appendingPathComponent("request.json")
        try Data("{}".utf8).write(to: path)
        XCTAssertThrowsError(try owner.prepareLaunch())
        try request.encoded().write(to: path)
        try Data("{}".utf8).write(to: owner.directory.appendingPathComponent("claim.json"))
        XCTAssertThrowsError(try owner.prepareLaunch())
        XCTAssertThrowsError(try WorkflowFileLease(directory: owner.directory, name: "activity.lock", mode: .shared))
    }
    func testNamespaceSymlinkAndOwnershipReplacementAreRejected() throws {
        let (root, request) = try fixture()
        let namespace = root.appendingPathComponent("workflow-operations")
        let outside = root.appendingPathComponent("outside")
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: false)
        try FileManager.default.createSymbolicLink(at: namespace, withDestinationURL: outside)
        XCTAssertThrowsError(try WorkflowOperationOwnership.create(workspace: root, request: request))
        try FileManager.default.removeItem(at: namespace)
        let owner = try WorkflowOperationOwnership.create(workspace: root, request: request)
        defer { owner.close() }
        try Data("{}".utf8).write(to: owner.directory.appendingPathComponent("ownership.json"))
        XCTAssertThrowsError(try owner.check())
    }
}
