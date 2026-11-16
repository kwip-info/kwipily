import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowRetainedHistoryReaderTests:XCTestCase {
    func testReceiptListingIsNotVerificationAndProjectsAreIsolated()async throws {
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let project=UUID().uuidString,operation=UUID().uuidString
        let directory=root.appendingPathComponent("workflow-operations/\(project)/\(operation)")
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        let document=WorkflowDocument(projectID:project,name:"PRIVATE SOURCE /secret/path")
        let receipt:JSONValue = .object(["schemaVersion":1,"projectID":.string(project),"operationID":.string(operation),"completedAt":.number(Date().timeIntervalSinceReferenceDate),"document":try JSONDecoder().decode(JSONValue.self,from:WorkflowCodec.encode(document)),"requestSHA256":.string(String(repeating:"b",count:64)),"manifestSHA256":.string(String(repeating:"a",count:64))])
        let bytes=try JSONEncoder().encode(receipt)
        try bytes.write(to:directory.appendingPathComponent("preparation-history.json"))
        let listing=try await WorkflowRetainedHistoryReader.list(root:root,projectID:project,family:.preparation)
        let metadata=try XCTUnwrap(listing.records.first)
        XCTAssertEqual(metadata.verification,"recordedReceiptOnly")
        let projected=String(decoding:try JSONEncoder().encode(metadata),as:UTF8.self)
        XCTAssertFalse(projected.contains("PRIVATE"));XCTAssertFalse(projected.contains("/secret"));XCTAssertFalse(projected.contains("document"))
        do {_ = try await WorkflowRetainedHistoryReader.inspect(root:root,projectID:project,metadata:metadata);XCTFail("Receipt alone must not verify absent artifacts")} catch {}
        guard case .object(var backendReceipt)=receipt else {return XCTFail("Fixture is not an object")}
        backendReceipt["workerRequest"] = .string(Data("{\"backendID\":\"sklearn\"}".utf8).base64EncodedString())
        backendReceipt["outerRequestSHA256"] = backendReceipt.removeValue(forKey:"requestSHA256")
        try JSONEncoder().encode(JSONValue.object(backendReceipt)).write(to:directory.appendingPathComponent("backend-history.json"))
        let backendList=try await WorkflowRetainedHistoryReader.list(root:root,projectID:project,family:.backend)
        let backend=try XCTUnwrap(backendList.records.first)
        XCTAssertEqual(backend.family,.backend)
        do {_ = try await WorkflowRetainedHistoryReader.inspect(root:root,projectID:project,metadata:backend);XCTFail("Backend receipt alone must not verify")} catch {}
        let foreign=try await WorkflowRetainedHistoryReader.list(root:root,projectID:UUID().uuidString,family:.preparation)
        XCTAssertTrue(foreign.records.isEmpty)
        XCTAssertEqual(try Data(contentsOf:directory.appendingPathComponent("preparation-history.json")),bytes)
        try Data("invalid".utf8).write(to:directory.appendingPathComponent("preparation-history.json"))
        let corrupt=try await WorkflowRetainedHistoryReader.list(root:root,projectID:project,family:.preparation)
        XCTAssertTrue(corrupt.records.isEmpty);XCTAssertEqual(corrupt.issues.first?.code,"receipt_unavailable")
    }
    func testBoundsUnsafeScopeCancellationAndUnsupportedFamily()async throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        for (project,offset,limit) in [("../escape",0,50),(UUID().uuidString,-1,50),(UUID().uuidString,0,101)] {
            do {_ = try await WorkflowRetainedHistoryReader.list(root:root,projectID:project,family:.backend,offset:offset,limit:limit);XCTFail("Invalid scope accepted")} catch {}
        }
        XCTAssertNil(WorkflowRetainedHistoryReader.Family(rawValue:"training"))
        let task=Task {try await WorkflowRetainedHistoryReader.list(root:root,projectID:UUID().uuidString,family:.backend)}
        task.cancel()
        do {_ = try await task.value;XCTFail("Cancelled read returned success")} catch is CancellationError {} catch {XCTFail("Expected cancellation")}
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.path))
    }
    func testListingPaginationAndUnsafeDirectoryAreBounded()async throws {
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let project=UUID().uuidString
        let folder=root.appendingPathComponent("workflow-operations/\(project)")
        for _ in 0..<3 {try FileManager.default.createDirectory(at:folder.appendingPathComponent(UUID().uuidString),withIntermediateDirectories:true)}
        let snapshot=try await WorkflowRetainedHistoryReader.open(root:root,projectID:project,family:.backend)
        let first=try snapshot.page(limit:2)
        XCTAssertEqual(first.inspectedOperations,2);XCTAssertNotNil(first.nextCursor)
        try FileManager.default.createDirectory(at:folder.appendingPathComponent(UUID().uuidString),withIntermediateDirectories:true)
        let last=try snapshot.page(cursor:first.nextCursor,limit:2)
        let fresh=try await WorkflowRetainedHistoryReader.open(root:root,projectID:project,family:.backend)
        XCTAssertThrowsError(try fresh.page(cursor:first.nextCursor))
        XCTAssertEqual(last.inspectedOperations,1);XCTAssertNil(last.nextCursor)
        try FileManager.default.removeItem(at:folder)
        try FileManager.default.createSymbolicLink(at:folder,withDestinationURL:root)
        do {_ = try await WorkflowRetainedHistoryReader.list(root:root,projectID:project,family:.backend);XCTFail("Followed symlink")} catch {}
    }

    func testMalformedReceiptsConsumeAggregateBudgetAndNoPartialSnapshotEscapes()async throws {
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let project=UUID().uuidString
        for _ in 0..<3 {
            let folder=root.appendingPathComponent("workflow-operations/\(project)/\(UUID().uuidString)")
            try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
            try Data(repeating:32,count:4_000_000).write(to:folder.appendingPathComponent("preparation-history.json"))
        }
        do {_ = try await WorkflowRetainedHistoryReader.open(root:root,projectID:project,family:.preparation,metadataAllowance:8_000_000);XCTFail("Aggregate metadata budget bypassed")}
        catch WorkflowRetainedHistoryReader.ReadFailure.metadataBudgetExceeded {}
        catch {XCTFail("Unexpected error: \(error)")}
    }

    @MainActor func testRealBackendRetainedEvidenceAndTamperRefusal()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_SKLEARN_TEST_PYTHON"] else {throw XCTSkip("Select pinned sklearn interpreter")}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(PrebuiltModelsPractice.lessonID,device:"cpu"))
        let host=try XCTUnwrap(workspace.workflowSession)
        workspace.sklearnRuntimeCapability=try .decode(JSONEncoder().encode(SklearnRuntimeCapabilities.expected))
        host.executionLaunch={_,args,dir,line,exit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),args,dir,line,exit)}
        XCTAssertTrue(host.run(train:PrebuiltModelsPractice.estimator),host.statusMessage)
        let deadline=Date().addingTimeInterval(60)
        while host.hasActiveOperation && Date()<deadline {try await Task.sleep(for:.milliseconds(40))}
        XCTAssertEqual(host.backendFit?.phase,.completed,host.statusMessage)
        let originalDocument=host.owner.document,originalSelection=host.owner.selection
        let entry=try XCTUnwrap(host.backendHistory.entries.first)
        let snapshot=try await WorkflowRetainedHistoryReader.open(root:workspace.workspaceDirectory,projectID:host.owner.projectID,family:.backend)
        let metadata=try XCTUnwrap(try snapshot.page().records.first)
        let verified=try await WorkflowRetainedHistoryReader.inspect(root:workspace.workspaceDirectory,projectID:host.owner.projectID,metadata:metadata)
        XCTAssertEqual(verified.provenance.verification,"verifiedRetainedArtifacts")
        XCTAssertEqual(verified.provenance.artifactCount,entry.artifacts.count)
        let access=MCPHistoryReadAccess()
        let allow:(Set<String>,Bool)throws->Void={_,_ in}
        let listed=try await access.read(client:"history-test",project:host.owner.projectID,root:workspace.workspaceDirectory,args:["operation":.string("list"),"family":.string("backend")],includeValues:false,authorize:allow)
        let snapshotID=try XCTUnwrap(listed["snapshotID"])
        let inspection:[String:JSONValue]=["operation":.string("inspect"),"family":.string("backend"),"snapshotID":snapshotID,"recordID":.string(entry.id)]
        let inspected=try await access.read(client:"history-test",project:host.owner.projectID,root:workspace.workspaceDirectory,args:inspection,includeValues:true,authorize:allow)
        XCTAssertEqual(inspected["verification"],.string("verifiedRetainedArtifacts"))
        XCTAssertFalse(String(decoding:try JSONEncoder().encode(inspected),as:UTF8.self).contains(root.path))
        let receiptURL=entry.artifactDirectory.deletingLastPathComponent().appendingPathComponent("backend-history.json")
        let receiptBytes=try Data(contentsOf:receiptURL)
        var changedReceipt=try JSONDecoder().decode([String:JSONValue].self,from:receiptBytes)
        changedReceipt["completedAt"] = .number(entry.completedAt.timeIntervalSinceReferenceDate+1)
        try JSONEncoder().encode(changedReceipt).write(to:receiptURL)
        do {_=try await access.read(client:"history-test",project:host.owner.projectID,root:workspace.workspaceDirectory,args:inspection,includeValues:true,authorize:allow);XCTFail("Changed receipt accepted under old metadata")}
        catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"history_evidence_unavailable")}
        try receiptBytes.write(to:receiptURL)
        XCTAssertEqual(host.owner.document,originalDocument);XCTAssertEqual(host.owner.selection,originalSelection)
        let manifest=entry.artifactDirectory.appendingPathComponent("manifest.json")
        let bytes=try Data(contentsOf:manifest)
        try Data("tampered".utf8).write(to:manifest)
        do {_ = try await WorkflowRetainedHistoryReader.inspect(root:workspace.workspaceDirectory,projectID:host.owner.projectID,metadata:metadata);XCTFail("Tampered backend accepted")} catch {}
        XCTAssertEqual(try Data(contentsOf:manifest),Data("tampered".utf8),"Read must not repair evidence")
        try bytes.write(to:manifest)
        _ = try await WorkflowRetainedHistoryReader.inspect(root:workspace.workspaceDirectory,projectID:host.owner.projectID,metadata:metadata)
    }

}
