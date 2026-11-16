import XCTest
import Foundation
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class WorkflowEvaluationOwnershipTests:XCTestCase {
    private func fixture(leaf:String=UUID().uuidString)throws->URL {
        guard let path=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw NSError(domain:NSPOSIXErrorDomain,code:Int(errno))}
        defer {free(path)}
        let root=URL(fileURLWithPath:String(cString:path)).appendingPathComponent(leaf)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        var info=statfs()
        guard statfs(root.path,&info)==0 else {throw NSError(domain:NSPOSIXErrorDomain,code:Int(errno))}
        let type=withUnsafePointer(to:&info.f_fstypename) {$0.withMemoryRebound(to:CChar.self,capacity:16) {String(cString:$0)}}
        guard type=="apfs",info.f_flags & UInt32(MNT_LOCAL) != 0 else {throw XCTSkip("Requires local APFS ownership semantics.")}
        return root
    }
    private func request()->WorkflowEvaluationWorkerRequest {
        .init(evaluationID:UUID().uuidString,projectID:UUID().uuidString,requestIdentity:String(repeating:"a",count:64),snapshotReceiptSHA256:String(repeating:"b",count:64),settings:.init())
    }
    func testRetainedInspectionExcludesOwnerAndActivityUntilExplicitClose()throws {
        let root=try fixture(),request=request()
        let owner=try WorkflowEvaluationOwnership.create(workspace:root,request:request)
        try request.encoded().write(to:owner.directory.appendingPathComponent("request.json"))
        XCTAssertThrowsError(try WorkflowEvaluationOwnership.acquireInspection(directory:owner.directory,projectID:request.projectID,evaluationID:request.evaluationID))
        owner.close()
        let inspection=try WorkflowEvaluationOwnership.acquireInspection(directory:owner.directory,projectID:request.projectID,evaluationID:request.evaluationID)
        defer {inspection.close()}
        XCTAssertEqual(inspection.metadata,owner.metadata)
        XCTAssertEqual(inspection.directory,owner.directory)
        XCTAssertNoThrow(try inspection.check())
        for name in ["owner.lock","activity.lock"] {
            XCTAssertThrowsError(try WorkflowFileLease(directory:owner.directory,name:name,mode:.shared))
        }
        inspection.close();inspection.close()
        XCTAssertThrowsError(try inspection.check())
        XCTAssertEqual(try WorkflowEvaluationOwnership.probe(directory:owner.directory,request:request),.inactive)
    }
    func testRetainedInspectionRevalidatesMetadataAndReleasesOnDeinit()throws {
        let root=try fixture(),request=request()
        let owner=try WorkflowEvaluationOwnership.create(workspace:root,request:request)
        try request.encoded().write(to:owner.directory.appendingPathComponent("request.json"));owner.close()
        var inspection:WorkflowEvaluationInspection?=try WorkflowEvaluationOwnership.acquireInspection(directory:owner.directory,projectID:request.projectID,evaluationID:request.evaluationID)
        let path=owner.directory.appendingPathComponent("ownership.json"),original=try Data(contentsOf:path)
        try Data("{}".utf8).write(to:path)
        XCTAssertThrowsError(try inspection?.check())
        try original.write(to:path)
        XCTAssertNoThrow(try inspection?.check())
        inspection=nil
        XCTAssertEqual(try WorkflowEvaluationOwnership.probe(directory:owner.directory,request:request),.inactive)
    }
    func testFailedInspectionAcquisitionDoesNotLeakOwnerLock()throws {
        let root=try fixture(),request=request()
        let owner=try WorkflowEvaluationOwnership.create(workspace:root,request:request)
        owner.close()
        let activity=try WorkflowFileLease(directory:owner.directory,name:"activity.lock",mode:.shared)
        defer {activity.close()}
        XCTAssertThrowsError(try WorkflowEvaluationOwnership.acquireInspection(directory:owner.directory,projectID:request.projectID,evaluationID:request.evaluationID))
        let available=try WorkflowFileLease(directory:owner.directory,name:"owner.lock",mode:.exclusive)
        available.close()
    }
    func testCreationPinsExactRequestAndExclusiveOwnerUntilClose()throws {
        let root=try fixture(),request=request()
        let owner=try WorkflowEvaluationOwnership.create(workspace:root,request:request)
        defer {owner.close()}
        XCTAssertEqual(owner.directory.path,root.appendingPathComponent("workflow-evaluations").appendingPathComponent(request.projectID).appendingPathComponent(request.evaluationID).path)
        XCTAssertEqual(owner.metadata.schemaVersion,1)
        XCTAssertEqual(owner.metadata.kind,"frozenEvaluation")
        XCTAssertEqual(owner.metadata.evaluationID,request.evaluationID)
        XCTAssertEqual(owner.metadata.projectID,request.projectID)
        XCTAssertEqual(owner.metadata.requestIdentity,request.requestIdentity)
        XCTAssertEqual(owner.metadata.requestSHA256,try request.requestSHA256())
        XCTAssertNoThrow(try owner.check())
        XCTAssertNoThrow(try owner.validateDirectory(owner.directory))
        XCTAssertThrowsError(try owner.validateDirectory(root))
        XCTAssertEqual(try WorkflowEvaluationOwnership.probe(directory:owner.directory,request:request),.inUse)
        XCTAssertThrowsError(try WorkflowEvaluationOwnership.create(workspace:root,request:request))
        let bytes=try Data(contentsOf:owner.directory.appendingPathComponent("ownership.json"))
        try request.encoded().write(to:owner.directory.appendingPathComponent("request.json"))
        try owner.prepareLaunch()
        owner.close();owner.close()
        XCTAssertThrowsError(try owner.check())
        XCTAssertEqual(try WorkflowEvaluationOwnership.probe(directory:owner.directory,request:request),.inactive)
        XCTAssertEqual(try Data(contentsOf:owner.directory.appendingPathComponent("ownership.json")),bytes)
    }
    func testStagingRetainsActivityUntilExactRequestPublished()throws {
        let request=request(),root=try fixture(leaf:request.evaluationID)
        let owner=try WorkflowEvaluationOwnership.initialize(directory:root,request:request)
        defer {owner.close()}
        XCTAssertEqual(owner.ownershipSHA256,WorkflowArchiveCodec.hash(try Data(contentsOf:root.appendingPathComponent("ownership.json"))))
        XCTAssertNoThrow(try owner.check())
        XCTAssertThrowsError(try WorkflowFileLease(directory:root,name:"activity.lock",mode:.shared))
        XCTAssertThrowsError(try owner.prepareLaunch())
        try Data("{}".utf8).write(to:root.appendingPathComponent("request.json"))
        XCTAssertThrowsError(try owner.prepareLaunch())
        XCTAssertThrowsError(try WorkflowFileLease(directory:root,name:"activity.lock",mode:.shared))
        try request.encoded().write(to:root.appendingPathComponent("request.json"))
        try owner.prepareLaunch()
        let activity=try WorkflowFileLease(directory:root,name:"activity.lock",mode:.shared)
        defer {activity.close()}
        XCTAssertNoThrow(try owner.check())
        XCTAssertEqual(try WorkflowEvaluationOwnership.probe(directory:root,request:request),.inUse)
    }
    func testAbandonedPartialStagingIsUnavailableAndWrongLeafRejects()throws {
        let request=request(),root=try fixture(leaf:request.evaluationID)
        XCTAssertThrowsError(try WorkflowEvaluationOwnership.initialize(directory:try fixture(),request:request))
        let owner=try WorkflowEvaluationOwnership.initialize(directory:root,request:request)
        XCTAssertEqual(try WorkflowEvaluationOwnership.probe(directory:root,request:request),.inUse)
        owner.close()
        XCTAssertThrowsError(try WorkflowEvaluationOwnership.probe(directory:root,request:request))
    }
    func testExistingEmptyInitializationNeverOverwritesEvidence()throws {
        let request=request(),root=try fixture(leaf:request.evaluationID)
        let owner=try WorkflowEvaluationOwnership.initialize(directory:root,request:request)
        owner.close()
        let bytes=try Data(contentsOf:root.appendingPathComponent("ownership.json"))
        XCTAssertThrowsError(try WorkflowEvaluationOwnership.initialize(directory:root,request:request))
        XCTAssertEqual(try Data(contentsOf:root.appendingPathComponent("ownership.json")),bytes)
        let other=self.request(),populated=try fixture(leaf:other.evaluationID)
        try Data("evidence".utf8).write(to:populated.appendingPathComponent("request.json"))
        XCTAssertThrowsError(try WorkflowEvaluationOwnership.initialize(directory:populated,request:other))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:populated.path),["request.json"])
    }
    func testForeignRequestAndMetadataTamperingRejectEvenWhileOwnerHeld()throws {
        let request=request(),root=try fixture(leaf:request.evaluationID)
        let owner=try WorkflowEvaluationOwnership.initialize(directory:root,request:request)
        defer {owner.close()}
        let other=WorkflowEvaluationWorkerRequest(evaluationID:request.evaluationID,projectID:request.projectID,requestIdentity:request.requestIdentity,snapshotReceiptSHA256:request.snapshotReceiptSHA256,settings:.init(batchSize:17))
        XCTAssertThrowsError(try WorkflowEvaluationOwnership.probe(directory:root,request:other))
        let path=root.appendingPathComponent("ownership.json")
        var object=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:path)) as? [String:Any])
        object["unexpected"]=true
        try JSONSerialization.data(withJSONObject:object).write(to:path)
        XCTAssertThrowsError(try owner.check())
        XCTAssertThrowsError(try WorkflowEvaluationOwnership.probe(directory:root,request:request))
    }
    func testReplacedActivityOrOwnerLockInvalidatesAuthorityAndProbe()throws {
        for name in ["owner.lock","activity.lock"] {
            let request=request(),root=try fixture(leaf:request.evaluationID)
            let owner=try WorkflowEvaluationOwnership.initialize(directory:root,request:request)
            defer {owner.close()}
            let path=root.appendingPathComponent(name)
            try FileManager.default.moveItem(at:path,to:root.appendingPathComponent(name+".old"))
            try Data().write(to:path)
            XCTAssertThrowsError(try owner.check(),name)
            XCTAssertThrowsError(try WorkflowEvaluationOwnership.probe(directory:root,request:request),name)
        }
    }
    func testRedirectedNamespaceIsRejectedWithoutWritingThroughIt()throws {
        let root=try fixture(),outside=try fixture(),request=request()
        try FileManager.default.createSymbolicLink(at:root.appendingPathComponent("workflow-evaluations"),withDestinationURL:outside)
        XCTAssertThrowsError(try WorkflowEvaluationOwnership.create(workspace:root,request:request))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:outside.path),[])
    }
    func testSeparateProcessOwnerAndActivityPreventInactiveProbeUntilActualExit()throws {
        for (name,mode) in [("owner.lock","EX"),("activity.lock","SH")] {
            let request=request(),root=try fixture(leaf:request.evaluationID)
            let owner=try WorkflowEvaluationOwnership.initialize(directory:root,request:request)
            try request.encoded().write(to:root.appendingPathComponent("request.json"))
            try owner.prepareLaunch()
            owner.close()
            let before=try Data(contentsOf:root.appendingPathComponent("ownership.json"))
            let process=Process(),input=Pipe(),output=Pipe()
            process.executableURL=URL(fileURLWithPath:"/usr/bin/python3")
            process.arguments=["-c","""
import fcntl, os, signal, sys
signal.alarm(10)
fd=os.open(sys.argv[1],os.O_RDWR)
fcntl.flock(fd, getattr(fcntl, 'LOCK_'+sys.argv[2])|fcntl.LOCK_NB)
os.write(1,b'READY\\n')
os.read(0,1)
os.close(fd)
""",root.appendingPathComponent(name).path,mode]
            process.standardInput=input;process.standardOutput=output;process.standardError=FileHandle.nullDevice
            try process.run()
            defer {
                try? input.fileHandleForWriting.close()
                if process.isRunning {process.terminate()}
                process.waitUntilExit()
            }
            XCTAssertEqual(output.fileHandleForReading.readData(ofLength:6),Data("READY\n".utf8))
            XCTAssertTrue(process.isRunning)
            XCTAssertEqual(try WorkflowEvaluationOwnership.probe(directory:root,request:request),.inUse,name)
            try input.fileHandleForWriting.close()
            process.waitUntilExit()
            XCTAssertEqual(process.terminationStatus,0)
            XCTAssertEqual(try WorkflowEvaluationOwnership.probe(directory:root,request:request),.inactive,name)
            XCTAssertEqual(try Data(contentsOf:root.appendingPathComponent("ownership.json")),before)
        }
    }
}
