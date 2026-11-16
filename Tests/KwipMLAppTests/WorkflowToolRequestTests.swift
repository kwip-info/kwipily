import XCTest
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class WorkflowToolRequestTests: XCTestCase {
    private let assessment = Data("{\"assessmentID\":\"test\"}".utf8)
    private let submission = Data("{\"answer\":42}".utf8)
    private func fixture(assessment: Bool = true) throws -> (URL, WorkflowToolRequest) {
        guard let canonical = realpath(FileManager.default.temporaryDirectory.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        defer { free(canonical) }
        let root = URL(fileURLWithPath: String(cString: canonical)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let project = UUID().uuidString, operation = UUID().uuidString
        let directory = root.appendingPathComponent("workflow-operations/\(project)/\(operation)")
        let files = assessment ? [WorkflowArchiveFile(path: "input/assessment.json", bytes: self.assessment.count, sha256: WorkflowArchiveCodec.hash(self.assessment)),
                                 WorkflowArchiveFile(path: "input/submission.json", bytes: submission.count, sha256: WorkflowArchiveCodec.hash(submission))] : []
        return (root, .init(operationID: operation, projectID: project, requestIdentity: String(repeating: "a", count: 64),
            toolMode: assessment ? .learningAssessment : .learningRuntime,
            arguments: assessment ? ["--learning-evidence-request", directory.appendingPathComponent("input/assessment.json").path,
                "--learning-submission", directory.appendingPathComponent("input/submission.json").path,
                "--learning-submission-sha256", WorkflowArchiveCodec.hash(submission)] : ["--learning-runtime"],
            inputFiles: files, executionBudget: try .resolve(global: .factory, projectMaximumRunSeconds: 60), remainingSeconds: assessment ? 30 : 10))
    }
    private func stage(_ owner: WorkflowOperationOwner) throws {
        try DiagnosticFileIO.writeImmutable(assessment, root: owner.directory, parts: ["input", "assessment.json"], limit: 2_000_000)
        try DiagnosticFileIO.writeImmutable(submission, root: owner.directory, parts: ["input", "submission.json"], limit: 2_000_000)
    }
    func testCodeExportPinsOnlyItsGraphAndEnforcesThirtySecondCeiling() throws {
        let (root, runtime) = try fixture(assessment: false)
        let directory = root.appendingPathComponent("workflow-operations/\(runtime.projectID)/\(runtime.operationID)")
        let graph = Data("{\"schemaVersion\":1}".utf8)
        let request = WorkflowToolRequest(operationID: runtime.operationID, projectID: runtime.projectID,
            requestIdentity: runtime.requestIdentity, toolMode: .codeExport,
            arguments: ["--export-code", "--graph", directory.appendingPathComponent("input/graph.json").path],
            inputFiles: [.init(path: "input/graph.json", bytes: graph.count, sha256: WorkflowArchiveCodec.hash(graph))],
            executionBudget: runtime.executionBudget, remainingSeconds: 30)
        XCTAssertEqual(try WorkflowToolRequest.decode(request.encoded()), request)
        let original = try XCTUnwrap(JSONSerialization.jsonObject(with: request.encoded()) as? [String: Any])
        let mutations: [(inout [String: Any]) -> Void] = [
            { $0["remainingSeconds"] = 30.01 },
            { $0["arguments"] = request.arguments + ["--registry", "/tmp/registry.json"] },
            { $0["arguments"] = ["--export-code", "--graph", "/tmp/graph.json"] },
            { $0["inputFiles"] = [] },
            { var files = $0["inputFiles"] as! [[String: Any]]; files[0]["bytes"] = 2_000_001; $0["inputFiles"] = files }
        ]
        for mutate in mutations {
            var object = original; mutate(&object)
            XCTAssertThrowsError(try WorkflowToolRequest.decode(JSONSerialization.data(withJSONObject: object)))
        }
        let owner = try WorkflowOperationOwnership.create(workspace: root, request: request)
        defer { owner.close() }
        XCTAssertThrowsError(try owner.publishRequest(request))
        try DiagnosticFileIO.writeImmutable(graph, root: owner.directory, parts: ["input", "graph.json"], limit: 2_000_000)
        try owner.publishRequest(request)
        XCTAssertEqual(try owner.prepareLaunch(), owner.ownershipSHA256)
    }
    func testWorkflowCodeExportPinsOnlyItsSnapshotAndRejectsCrossedGraphFlags() throws {
        let (root, runtime) = try fixture(assessment: false)
        let directory = root.appendingPathComponent("workflow-operations/\(runtime.projectID)/\(runtime.operationID)")
        let snapshot = Data("{\"schemaVersion\":1,\"scope\":\"workflow\"}".utf8)
        let request = WorkflowToolRequest(operationID: runtime.operationID, projectID: runtime.projectID,
            requestIdentity: runtime.requestIdentity, toolMode: .codeExport,
            arguments: ["--export-workflow-code", "--workflow-code-plan", directory.appendingPathComponent("input/workflow-code.json").path],
            inputFiles: [.init(path: "input/workflow-code.json", bytes: snapshot.count, sha256: WorkflowArchiveCodec.hash(snapshot))],
            executionBudget: runtime.executionBudget, remainingSeconds: 30)
        XCTAssertEqual(try WorkflowToolRequest.decode(request.encoded()), request)
        let original = try XCTUnwrap(JSONSerialization.jsonObject(with: request.encoded()) as? [String: Any])
        let mutations: [(inout [String: Any]) -> Void] = [
            { $0["arguments"] = ["--export-code", "--graph", request.arguments[2]] },
            { $0["arguments"] = request.arguments + ["--source-bindings", "/tmp/private.json"] },
            { $0["arguments"] = ["--export-workflow-code", "--workflow-code-plan", "/tmp/workflow-code.json"] },
            { var files = $0["inputFiles"] as! [[String: Any]]; files[0]["path"] = "input/graph.json"; $0["inputFiles"] = files },
            { var files = $0["inputFiles"] as! [[String: Any]]; files.append(files[0]); $0["inputFiles"] = files },
            { $0["remainingSeconds"] = 30.01 }
        ]
        for mutate in mutations {
            var object = original; mutate(&object)
            XCTAssertThrowsError(try WorkflowToolRequest.decode(JSONSerialization.data(withJSONObject: object)))
        }
        let owner = try WorkflowOperationOwnership.create(workspace: root, request: request)
        defer { owner.close() }
        try DiagnosticFileIO.writeImmutable(snapshot, root: owner.directory, parts: ["input", "workflow-code.json"], limit: 2_000_000)
        try owner.publishRequest(request)
        try Data("{}".utf8).write(to: owner.directory.appendingPathComponent("input/workflow-code.json"))
        XCTAssertThrowsError(try owner.prepareLaunch())
        try snapshot.write(to: owner.directory.appendingPathComponent("input/workflow-code.json"))
        XCTAssertEqual(try owner.prepareLaunch(), owner.ownershipSHA256)
    }
    func testExactToolRequestsRoundTripAndMalformedFieldsReject() throws {
        for assessment in [false, true] {
            let (_, request) = try fixture(assessment: assessment)
            XCTAssertEqual(try WorkflowToolRequest.decode(request.encoded()), request)
            let original = try XCTUnwrap(JSONSerialization.jsonObject(with: request.encoded()) as? [String: Any])
            let mutations: [(inout [String: Any]) -> Void] = [
                { $0["schemaVersion"] = 1 }, { $0["phase"] = "training" }, { $0["toolMode"] = "shell" },
                { $0["remainingSeconds"] = 61 }, { $0["remainingSeconds"] = 0 }, { $0["remainingSeconds"] = true },
                { $0.removeValue(forKey: "executionBudget") }, { $0["arguments"] = ["--capabilities"] },
                { var policy = $0["executionBudget"] as! [String: Any]; policy["capabilityMaximumSeconds"] = NSNull(); $0["executionBudget"] = policy }
            ]
            for mutate in mutations {
                var object = original; mutate(&object)
                XCTAssertThrowsError(try WorkflowToolRequest.decode(JSONSerialization.data(withJSONObject: object)))
            }
        }
    }
    func testOwnerRequiresStagedPinnedInputsAndNeverReleasesOnTamper() throws {
        let (root, request) = try fixture()
        let owner = try WorkflowOperationOwnership.create(workspace: root, request: request)
        defer { owner.close() }
        XCTAssertThrowsError(try owner.publishRequest(request))
        try stage(owner); try owner.publishRequest(request)
        let path = owner.directory.appendingPathComponent("input/submission.json")
        try Data("changed".utf8).write(to: path)
        XCTAssertThrowsError(try owner.prepareLaunch())
        XCTAssertThrowsError(try WorkflowFileLease(directory: owner.directory, name: "activity.lock", mode: .shared))
        try submission.write(to: path)
        XCTAssertEqual(try owner.prepareLaunch(), owner.ownershipSHA256)
        XCTAssertThrowsError(try owner.prepareLaunch())
    }
    func testRuntimeUsesSameOwnerWithoutInputFilesAndRetainsOwnerUntilExit() throws {
        let (root, request) = try fixture(assessment: false)
        let owner = try WorkflowOperationOwnership.create(workspace: root, request: request)
        try owner.publishRequest(request); try owner.prepareLaunch()
        XCTAssertThrowsError(try WorkflowFileLease(directory: owner.directory, name: "owner.lock", mode: .shared))
        owner.close()
        let lease = try WorkflowFileLease(directory: owner.directory, name: "owner.lock", mode: .exclusive)
        lease.close()
    }
    func testWrongScopeSymlinkInputsAndMalformedInventoryReject() throws {
        let (root, request) = try fixture()
        XCTAssertThrowsError(try WorkflowOperationOwnership.create(workspace: root.appendingPathComponent("other"), request: request))
        let owner = try WorkflowOperationOwnership.create(workspace: root, request: request)
        defer { owner.close() }
        try stage(owner)
        let source = owner.directory.appendingPathComponent("input/submission.json")
        let moved = root.appendingPathComponent("submission.json")
        try FileManager.default.moveItem(at: source, to: moved)
        try FileManager.default.createSymbolicLink(at: source, withDestinationURL: moved)
        XCTAssertThrowsError(try owner.publishRequest(request))
        let original = try XCTUnwrap(JSONSerialization.jsonObject(with: request.encoded()) as? [String: Any])
        for change: (inout [[String: Any]]) -> Void in [
            { $0[0]["bytes"] = 2_000_001 }, { $0[0]["path"] = "../assessment.json" },
            { $0[0]["extra"] = true }, { $0.append($0[0]) }, { $0[1]["sha256"] = String(repeating: "f", count: 64) }
        ] {
            var object = original, files = original["inputFiles"] as! [[String: Any]]
            change(&files); object["inputFiles"] = files
            XCTAssertThrowsError(try WorkflowToolRequest.decode(JSONSerialization.data(withJSONObject: object)))
        }
    }
}
