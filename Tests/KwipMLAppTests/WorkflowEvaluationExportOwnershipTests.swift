import XCTest
import KwipMLCore
import Darwin
@testable import KwipMLApp

final class WorkflowEvaluationExportOwnershipTests: XCTestCase {
    private func fixture() throws -> (URL, WorkflowEvaluationExportRequest) {
        guard let canonical = realpath(FileManager.default.temporaryDirectory.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        defer { free(canonical) }
        let root = URL(fileURLWithPath: String(cString: canonical)).appendingPathComponent(UUID().uuidString)
        let project = UUID().uuidString, evaluation = UUID().uuidString
        let source = root.appendingPathComponent("workflow-evaluations/\(project)/\(evaluation)")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return (root, .init(exportID: UUID().uuidString, projectID: project, evaluationID: evaluation,
            requestIdentity: String(repeating: "a", count: 64), evaluationDirectory: source.path,
            evaluationManifestSHA256: String(repeating: "b", count: 64), includeTestRows: false))
    }
    func testExactRequestRoundTripAndRejectsUnknownKeysAndWrongSourceScope() throws {
        let (_, request) = try fixture()
        XCTAssertEqual(try WorkflowEvaluationExportRequest.decode(request.encoded()), request)
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: request.encoded()) as? [String: Any])
        object["extra"] = true
        XCTAssertThrowsError(try WorkflowEvaluationExportRequest.decode(JSONSerialization.data(withJSONObject: object)))
        object.removeValue(forKey: "extra")
        object["projectID"] = UUID().uuidString
        XCTAssertThrowsError(try WorkflowEvaluationExportRequest.decode(JSONSerialization.data(withJSONObject: object)))
        object["projectID"] = request.projectID
        object["includeTestRows"] = 1
        XCTAssertThrowsError(try WorkflowEvaluationExportRequest.decode(JSONSerialization.data(withJSONObject: object)))
    }
    func testCreatePinsRequestHoldsOwnerAndReleasesActivityOnlyForOneLaunch() throws {
        let (root, request) = try fixture()
        let owner = try WorkflowEvaluationExportOwnership.create(workspace: root, request: request)
        defer { owner.close() }
        XCTAssertEqual(owner.directory.path, root.appendingPathComponent("workflow-evaluation-exports/\(request.projectID)/\(request.exportID)").path)
        XCTAssertEqual(owner.metadata.kind, "portableEvaluationExport")
        XCTAssertEqual(owner.metadata.requestSHA256, try request.requestSHA256())
        XCTAssertThrowsError(try WorkflowFileLease(directory: owner.directory, name: "owner.lock", mode: .shared))
        XCTAssertThrowsError(try WorkflowFileLease(directory: owner.directory, name: "activity.lock", mode: .shared))
        XCTAssertThrowsError(try owner.prepareLaunch())
        try owner.publishRequest(request)
        XCTAssertEqual(try owner.prepareLaunch(), owner.ownershipSHA256)
        let activity = try WorkflowFileLease(directory: owner.directory, name: "activity.lock", mode: .shared)
        activity.close()
        XCTAssertThrowsError(try owner.prepareLaunch())
        XCTAssertThrowsError(try WorkflowEvaluationExportOwnership.create(workspace: root, request: request))
        owner.close(); owner.close()
        XCTAssertThrowsError(try owner.check())
        let released = try WorkflowFileLease(directory: owner.directory, name: "owner.lock", mode: .exclusive)
        released.close()
    }
    func testChangedRequestOrExistingClaimCannotLaunch() throws {
        let (root, request) = try fixture()
        let owner = try WorkflowEvaluationExportOwnership.create(workspace: root, request: request)
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
    func testMetadataReplacementAndNamespaceSymlinkAreRejected() throws {
        let (root, request) = try fixture()
        let namespace = root.appendingPathComponent("workflow-evaluation-exports")
        let outside = root.appendingPathComponent("outside")
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: false)
        try FileManager.default.createSymbolicLink(at: namespace, withDestinationURL: outside)
        XCTAssertThrowsError(try WorkflowEvaluationExportOwnership.create(workspace: root, request: request))
        try FileManager.default.removeItem(at: namespace)
        let owner = try WorkflowEvaluationExportOwnership.create(workspace: root, request: request)
        defer { owner.close() }
        try Data("{}".utf8).write(to: owner.directory.appendingPathComponent("ownership.json"))
        XCTAssertThrowsError(try owner.check())
    }
}
