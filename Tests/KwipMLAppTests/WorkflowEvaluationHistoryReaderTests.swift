import XCTest
import Foundation
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class WorkflowEvaluationHistoryReaderTests:XCTestCase {
    private func fixture()throws->(URL,WorkflowEvaluationWorkerRequest,WorkflowEvaluationOwner) {
        guard let path=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw NSError(domain:NSPOSIXErrorDomain,code:Int(errno))}
        defer {free(path)}
        let root=URL(fileURLWithPath:String(cString:path)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let request=WorkflowEvaluationWorkerRequest(evaluationID:UUID().uuidString,projectID:UUID().uuidString,requestIdentity:String(repeating:"a",count:64),snapshotReceiptSHA256:String(repeating:"b",count:64),settings:.init())
        let owner=try WorkflowEvaluationOwnership.create(workspace:root,request:request)
        addTeardownBlock {owner.close()}
        return(root,request,owner)
    }
    private func reference(_ request:WorkflowEvaluationWorkerRequest)->WorkflowEvaluationHistoryReference {
        .init(projectID:request.projectID,evaluationID:UUID(uuidString:request.evaluationID)!)
    }
    private func outcome(_ request:WorkflowEvaluationWorkerRequest,status:WorkflowEvaluationStatus = .failed)->WorkflowEvaluationOutcome {
        .init(evaluationID:request.evaluationID,projectID:request.projectID,requestIdentity:request.requestIdentity,status:status,startedAt:Date(timeIntervalSince1970:100),finishedAt:Date(timeIntervalSince1970:99),capture:nil,manifestSHA256:nil,artifacts:[],error:status == .failed ? "Capture failed":nil)
    }
    private func write(_ value:WorkflowEvaluationOutcome,_ owner:WorkflowEvaluationOwner)throws {
        try WorkflowArchiveCodec.encode(value).write(to:owner.directory.appendingPathComponent("outcome.json"))
    }
    func testActivePartialStagingIsInUseThenAbandonedStagingIncomplete()throws {
        let(root,request,owner)=try fixture()
        guard case .inUse=WorkflowEvaluationHistoryReader.load(reference(request),root:root) else {return XCTFail("Active staging must not be interpreted as missing result")}
        owner.close()
        guard case .incomplete=WorkflowEvaluationHistoryReader.load(reference(request),root:root) else {return XCTFail("Abandoned staging has no terminal evidence")}
    }
    func testEarlyFailureAndCancellationAreRecordedWithCaptureLimitation()throws {
        for status in [WorkflowEvaluationStatus.failed,.cancelled] {
            let(root,request,owner)=try fixture()
            try write(outcome(request,status:status),owner);owner.close()
            guard case .terminal(let record)=WorkflowEvaluationHistoryReader.load(reference(request),root:root) else {XCTFail("Valid early terminal evidence should load");continue}
            XCTAssertEqual(record.outcome.status,status)
            XCTAssertTrue(record.captureIncomplete)
            XCTAssertNil(record.result)
            // Valid wall-clock values need not be monotonic across clock changes.
            XCTAssertLessThan(record.outcome.finishedAt,record.outcome.startedAt)
        }
    }
    func testRetainedInspectionReadsSameRecordWithoutReacquiringAndRejectsClosedOrForeignLease()throws {
        let(root,request,owner)=try fixture()
        try write(outcome(request,status:.cancelled),owner);owner.close()
        let inspection=try WorkflowEvaluationOwnership.acquireInspection(directory:owner.directory,projectID:request.projectID,evaluationID:request.evaluationID)
        defer {inspection.close()}
        guard case .inUse=WorkflowEvaluationHistoryReader.load(reference(request),root:root) else {return XCTFail("Ordinary reader must respect retained export ownership")}
        guard case .terminal(let record)=try WorkflowEvaluationHistoryReader.load(reference(request),inspection:inspection) else {return XCTFail("Retained reader must verify under existing ownership")}
        XCTAssertEqual(record.outcome.status,.cancelled)
        let foreign=WorkflowEvaluationHistoryReference(projectID:request.projectID,evaluationID:UUID())
        XCTAssertThrowsError(try WorkflowEvaluationHistoryReader.load(foreign,inspection:inspection))
        inspection.close()
        XCTAssertThrowsError(try WorkflowEvaluationHistoryReader.load(reference(request),inspection:inspection))
        guard case .terminal=WorkflowEvaluationHistoryReader.load(reference(request),root:root) else {return XCTFail("Close must release retained reader")}
    }
    func testHistoryListsRecordedFinishTimeWithoutDisclosingMetrics()throws {
        let(root,request,owner)=try fixture()
        let value=outcome(request,status:.cancelled)
        try write(value,owner)
        try request.encoded().write(to:owner.directory.appendingPathComponent("request.json"));owner.close()
        let newer=WorkflowEvaluationWorkerRequest(evaluationID:UUID().uuidString,projectID:request.projectID,requestIdentity:request.requestIdentity,snapshotReceiptSHA256:request.snapshotReceiptSHA256,settings:.init(partitionRole:"validation"))
        let other=try WorkflowEvaluationOwnership.create(workspace:root,request:newer)
        defer {other.close()}
        let later=WorkflowEvaluationOutcome(evaluationID:newer.evaluationID,projectID:newer.projectID,requestIdentity:newer.requestIdentity,status:.cancelled,startedAt:Date(timeIntervalSince1970:150),finishedAt:Date(timeIntervalSince1970:200),capture:nil,manifestSHA256:nil,artifacts:[],error:nil)
        try write(later,other)
        try newer.encoded().write(to:other.directory.appendingPathComponent("request.json"));other.close()
        let listing=try WorkflowEvaluationHistoryReader.list(root:root,projectID:request.projectID)
        XCTAssertEqual(listing.entries.count,2)
        XCTAssertEqual(listing.entries.map(\.finishedAt),[later.finishedAt,value.finishedAt])
        XCTAssertEqual(listing.entries.first?.status,.cancelled)
        XCTAssertEqual(listing.entries.map(\.partitionRole),["validation","test"])
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("workflow-evaluation-exposure").path),"Listing roles must not disclose or record metrics")
    }
    func testCompletePredictionInventoryListsWithoutDisclosingUnverifiedMetrics()throws {
        for extra in ["prediction-table.json","unexpected.json"] {
            let(root,request,owner)=try fixture()
            let files=["manifest.json","summary.json","predictions.jsonl",extra].map {WorkflowArchiveFile(path:"artifacts/"+$0,bytes:2,sha256:String(repeating:"c",count:64))}
            let value=WorkflowEvaluationOutcome(evaluationID:request.evaluationID,projectID:request.projectID,requestIdentity:request.requestIdentity,status:.completed,startedAt:Date(timeIntervalSince1970:100),finishedAt:Date(timeIntervalSince1970:101),capture:.init(path:"capture.json",bytes:2,sha256:String(repeating:"d",count:64)),manifestSHA256:String(repeating:"e",count:64),artifacts:files,error:nil)
            try write(value,owner);owner.close()
            let entry=try XCTUnwrap(WorkflowEvaluationHistoryReader.list(root:root,projectID:request.projectID).entries.first)
            XCTAssertEqual(entry.status,extra=="prediction-table.json" ? .completed:nil)
            guard case .unavailable=WorkflowEvaluationHistoryReader.load(reference(request),root:root) else {return XCTFail("A list receipt cannot substitute for pinned capture verification")}
        }
    }
    func testCompletedWithoutCaptureOrArtifactsIsUnavailable()throws {
        let(root,request,owner)=try fixture()
        try write(outcome(request,status:.completed),owner);owner.close()
        guard case .unavailable=WorkflowEvaluationHistoryReader.load(reference(request),root:root) else {return XCTFail("Cannot manufacture completed evidence")}
    }
    func testMalformedOrForeignOutcomeCannotBecomeTerminal()throws {
        for mutation in ["extra","identity","artifacts","time"] {
            let(root,request,owner)=try fixture()
            var object=try XCTUnwrap(JSONSerialization.jsonObject(with:WorkflowArchiveCodec.encode(outcome(request))) as? [String:Any])
            switch mutation {
            case "extra":object["unexpected"]=true
            case "identity":object["projectID"]=UUID().uuidString
            case "artifacts":object["artifacts"]=[["path":"artifacts/summary.json","bytes":1,"sha256":String(repeating:"a",count:64)]]
            default:object["finishedAt"]=1e100
            }
            try JSONSerialization.data(withJSONObject:object).write(to:owner.directory.appendingPathComponent("outcome.json"));owner.close()
            guard case .unavailable=WorkflowEvaluationHistoryReader.load(reference(request),root:root) else {XCTFail("Accepted invalid outcome: \(mutation)");continue}
        }
    }
    func testPartialCaptureClaimMustBeVerifiedRatherThanDowngradedToEarlyFailure()throws {
        let(root,request,owner)=try fixture()
        let initial=outcome(request)
        let value=WorkflowEvaluationOutcome(evaluationID:initial.evaluationID,projectID:initial.projectID,requestIdentity:initial.requestIdentity,status:.failed,startedAt:initial.startedAt,finishedAt:initial.finishedAt,capture:.init(path:"capture.json",bytes:2,sha256:WorkflowArchiveCodec.hash(Data("{}".utf8))),manifestSHA256:nil,artifacts:[],error:"Worker failed")
        try write(value,owner)
        try Data("{}".utf8).write(to:owner.directory.appendingPathComponent("capture.json"));owner.close()
        guard case .unavailable=WorkflowEvaluationHistoryReader.load(reference(request),root:root) else {return XCTFail("Declared capture must independently verify")}
    }
    func testListingRecordsOnlyStatusAndReportsTruncation()throws {
        let(root,request,owner)=try fixture()
        try write(outcome(request),owner);owner.close()
        let listing=try WorkflowEvaluationHistoryReader.list(root:root,projectID:request.projectID,limit:10)
        XCTAssertEqual(listing.entries.count,1)
        XCTAssertEqual(listing.entries.first?.status,.failed)
        XCTAssertEqual(listing.entries.first?.verification,.recordedUnverified)
        let second=WorkflowEvaluationWorkerRequest(evaluationID:UUID().uuidString,projectID:request.projectID,requestIdentity:request.requestIdentity,snapshotReceiptSHA256:request.snapshotReceiptSHA256,settings:.init())
        let secondOwner=try WorkflowEvaluationOwnership.create(workspace:root,request:second);secondOwner.close()
        let bounded=try WorkflowEvaluationHistoryReader.list(root:root,projectID:request.projectID,limit:1)
        XCTAssertEqual(bounded.entries.count,1)
        XCTAssertTrue(bounded.truncated)
        XCTAssertFalse(bounded.issues.isEmpty)
    }
    func testRedirectedEvaluationDirectoryIsUnavailableWithoutReadingTarget()throws {
        let(root,request,owner)=try fixture()
        try write(outcome(request),owner);owner.close()
        let saved=owner.directory.appendingPathExtension("saved")
        try FileManager.default.moveItem(at:owner.directory,to:saved)
        try FileManager.default.createSymbolicLink(at:owner.directory,withDestinationURL:saved)
        guard case .unavailable=WorkflowEvaluationHistoryReader.load(reference(request),root:root) else {return XCTFail("Symlink evaluation directory must not be followed")}
    }
    func testResealedCapturedFailureRejectsMissingPreparationAndFalseParentAuthority()throws {
        for mutation in ["omittedPreparation","outcomePin","parentIdentity"] {
            let(root,seed,unused)=try fixture();unused.close()
            let hash=String(repeating:"a",count:64),parentID=UUID().uuidString,runID=UUID().uuidString
            func json(_ object:Any)throws->Data {try JSONSerialization.data(withJSONObject:object,options:[.sortedKeys])}
            let preparation=try json(["artifacts":[["path":"missing-state.json","kind":"state","bytes":2,"sha256":WorkflowArchiveCodec.hash(Data("{}".utf8))]]])
            let training=Data("{}".utf8),checkpoint=Data("checkpoint".utf8)
            let parent:[String:Any]=["status":"completed","projectID":seed.projectID,"directoryID":parentID,"runID":runID,"requestIdentity":hash,"modelManifestSHA256":WorkflowArchiveCodec.hash(training),"preparationManifestSHA256":WorkflowArchiveCodec.hash(preparation)]
            let parentBytes=try json(parent)
            var captureParent=parent
            if mutation=="parentIdentity" {captureParent["runID"]=UUID().uuidString}
            let reference:[String:Any]=["projectID":seed.projectID,"directoryID":parentID,"runID":runID,"requestIdentity":hash,"outcomeSHA256":mutation=="outcomePin" ? hash:WorkflowArchiveCodec.hash(parentBytes),"modelManifestSHA256":WorkflowArchiveCodec.hash(training),"preparationManifestSHA256":WorkflowArchiveCodec.hash(preparation),"checkpointSHA256":WorkflowArchiveCodec.hash(checkpoint)]
            var contents:[String:Data]=["outcome.json":parentBytes,"capture.json":try json(captureParent),"workflow.json":Data("{}".utf8),"model-registry.json":Data("{}".utf8),"data-registry.json":Data("{}".utf8),"runtime-evidence.json":Data("{}".utf8),"training/artifacts/manifest.json":training,"training/artifacts/checkpoint.pt":checkpoint,"preparation/artifacts/manifest.json":preparation]
            if mutation != "omittedPreparation" {contents["preparation/artifacts/missing-state.json"]=Data("{}".utf8)}
            func file(_ path:String,_ bytes:Data)->[String:Any] {["path":path,"bytes":bytes.count,"sha256":WorkflowArchiveCodec.hash(bytes)]}
            let receipt=try json(["schemaVersion":1,"reference":reference,"testEndpoint":["node":"split","port":"test"],"files":contents.map{file($0.key,$0.value)}])
            let request=WorkflowEvaluationWorkerRequest(evaluationID:UUID().uuidString,projectID:seed.projectID,requestIdentity:hash,snapshotReceiptSHA256:WorkflowArchiveCodec.hash(receipt),settings:.init())
            let owner=try WorkflowEvaluationOwnership.create(workspace:root,request:request)
            defer {owner.close()}
            for (path,bytes) in contents {
                let target=owner.directory.appendingPathComponent("input/"+path)
                try FileManager.default.createDirectory(at:target.deletingLastPathComponent(),withIntermediateDirectories:true)
                try bytes.write(to:target)
            }
            try receipt.write(to:owner.directory.appendingPathComponent("input/snapshot.json"))
            let requestBytes=try request.encoded(),workflow=try WorkflowCodec.encode(WorkflowExamples.preparedGroupSplitClassifier(projectID:seed.projectID)),runtime=Data("{}".utf8)
            try requestBytes.write(to:owner.directory.appendingPathComponent("request.json"))
            try workflow.write(to:owner.directory.appendingPathComponent("workflow.json"))
            try runtime.write(to:owner.directory.appendingPathComponent("runtime-evidence.json"))
            let capture=try json(["schemaVersion":1,"kind":"frozenEvaluationCapture","evaluationID":request.evaluationID,"requestIdentity":hash,"bindingRevision":0,"workerRequest":file("request.json",requestBytes),"workflow":file("workflow.json",workflow),"runtimeEvidence":file("runtime-evidence.json",runtime),"snapshotReceipt":file("input/snapshot.json",receipt)])
            try capture.write(to:owner.directory.appendingPathComponent("capture.json"))
            let value=WorkflowEvaluationOutcome(evaluationID:request.evaluationID,projectID:request.projectID,requestIdentity:hash,status:.failed,startedAt:Date(),finishedAt:Date(),capture:.init(path:"capture.json",bytes:capture.count,sha256:WorkflowArchiveCodec.hash(capture)),manifestSHA256:nil,artifacts:[],error:"Worker failed")
            try write(value,owner);owner.close()
            guard case .unavailable=WorkflowEvaluationHistoryReader.load(self.reference(request),root:root) else {XCTFail("Accepted resealed invalid parent: \(mutation)");continue}
        }
    }

}
