import XCTest
import Foundation
import Darwin
import KwipMLCore
@testable import KwipMLApp

@MainActor final class BackendFitHistoryStoreTests:XCTestCase {
    func testHistoryRejectsInvalidProjectAndMissingOperation()async throws {
        let root=try canonicalBackendTemporaryDirectory()
        let store=BackendFitHistoryStore(root:root,projectID:"../escape")
        await store.reloadAndWait();XCTAssertTrue(store.entries.isEmpty);XCTAssertFalse(store.status.isEmpty)
        let valid=BackendFitHistoryStore(root:root,projectID:UUID().uuidString)
        do {try await valid.record(operationID:UUID().uuidString,workerRequest:Data(),document:WorkflowDocument(projectID:UUID().uuidString));XCTFail("Missing operation must be rejected")} catch {}
    }
    func testCancelledRecordStopsBeforeEvidenceIO()async throws {
        let root=try canonicalBackendTemporaryDirectory()
        let project=UUID().uuidString
        let store=BackendFitHistoryStore(root:root,projectID:project)
        let task=Task {try await store.record(operationID:UUID().uuidString,workerRequest:Data(),document:WorkflowDocument(projectID:project))}
        task.cancel()
        do {try await task.value;XCTFail("Cancelled History acceptance must not succeed")}
        catch is CancellationError {}
        catch {XCTFail("Cancellation should precede evidence IO: \(String(reflecting:error))")}
        XCTAssertTrue(store.entries.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.path))
    }
    func testRealArtifactRoundTripAndTamperHidesSuccess()async throws {
        guard let fixture=ProcessInfo.processInfo.environment["KWIPML_TENSORFLOW_FIXTURE"] else {throw XCTSkip("Requires actual TensorFlow artifact fixture.")}
        let root=try canonicalBackendTemporaryDirectory()
        defer {try? FileManager.default.removeItem(at:root)}
        let project=UUID().uuidString,operation=UUID().uuidString
        let directory=root.appendingPathComponent("workflow-operations/\(project)/\(operation)")
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        try FileManager.default.copyItem(at:URL(fileURLWithPath:fixture),to:directory.appendingPathComponent("artifacts"))
        let artifact=directory.appendingPathComponent("artifacts")
        let raw=try Data(contentsOf:artifact.appendingPathComponent("manifest.json"))
        let object=try XCTUnwrap(JSONSerialization.jsonObject(with:raw) as? [String:Any]);let p=try XCTUnwrap(object["provenance"] as? [String:Any]);let prep=try XCTUnwrap(p["preparation"] as? [String:Any])
        let configuration=try JSONDecoder().decode(TensorFlowPreparedWorkerRequest.Configuration.self,from:JSONSerialization.data(withJSONObject:try XCTUnwrap(object["configuration"])))
        let selection=PreparedTrainingSelection(manifestPath:directory.appendingPathComponent("input/preparation/manifest.json").path,manifestSHA256:try XCTUnwrap(prep["manifestSHA256"] as? String),graphDigest:try XCTUnwrap(prep["graphDigest"] as? String),trainingNode:"training",trainingPort:"table",validationNode:"heldout",validationPort:"table",featureColumns:["feature","second"],targetColumn:"group")
        let request=try TensorFlowPreparedWorkerRequest(graph:configuration.graph,settings:configuration.settings,selection:selection)
        let requestBytes=try JSONEncoder().encode(request)
        let outer=WorkflowToolRequest(operationID:operation,projectID:project,requestIdentity:String(repeating:"a",count:64),toolMode:.backendFit,arguments:["--fit-backend",directory.appendingPathComponent("input/backend-request.json").path,"--backend-request-sha256",WorkflowArchiveCodec.hash(requestBytes),"--output",artifact.path],inputFiles:[.init(path:"input/backend-request.json",bytes:requestBytes.count,sha256:WorkflowArchiveCodec.hash(requestBytes)),.init(path:"input/preparation/manifest.json",bytes:try Data(contentsOf:artifact.appendingPathComponent("preparation/manifest.json")).count,sha256:selection.manifestSHA256)],executionBudget:try .resolve(global:.factory,projectMaximumRunSeconds:60),remainingSeconds:60)
        try outer.encoded().write(to:directory.appendingPathComponent("request.json"))
        let document=WorkflowDocument(projectID:project)
        let store=BackendFitHistoryStore(root:root,projectID:project)
        do {try await store.record(operationID:operation,workerRequest:requestBytes,document:document)} catch {XCTFail("History record failed: \(String(reflecting:error))");return}
        XCTAssertEqual(store.entries.count,1,store.status)
        XCTAssertEqual(store.entries.first?.backendID,"tensorflow")
        let reopened=BackendFitHistoryStore(root:root,projectID:project);await reopened.reloadAndWait()
        XCTAssertEqual(reopened.entries.first?.id,operation,reopened.status)
        try Data("tampered".utf8).write(to:artifact.appendingPathComponent("portable/inference.py"))
        await reopened.reloadAndWait();XCTAssertTrue(reopened.entries.isEmpty);XCTAssertFalse(reopened.status.isEmpty)
    }
}

func canonicalBackendTemporaryDirectory()throws->URL {
    guard let raw=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
    defer {free(raw)}
    return URL(fileURLWithPath:String(cString:raw)).appendingPathComponent(UUID().uuidString)
}
