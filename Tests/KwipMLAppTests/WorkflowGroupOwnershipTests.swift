import XCTest
import Foundation
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class WorkflowGroupOwnershipTests:XCTestCase {
    private func fixture()throws->URL {
        guard let canonical=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw NSError(domain:NSPOSIXErrorDomain,code:Int(errno))}
        defer {free(canonical)}
        let directory=URL(fileURLWithPath:String(cString:canonical),isDirectory:true).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        var info=statfs()
        guard statfs(directory.path,&info) == 0 else {throw NSError(domain:NSPOSIXErrorDomain,code:Int(errno))}
        let type=withUnsafePointer(to:&info.f_fstypename) {pointer in pointer.withMemoryRebound(to:CChar.self,capacity:16) {String(cString:$0)}}
        guard type == "apfs",info.f_flags & UInt32(MNT_LOCAL) != 0 else {throw XCTSkip("Ownership integration requires a verified local APFS fixture.")}
        return directory
    }
    func testCurrentOwnerAndWorkerActivityExcludeRecoveryThenRestartSucceeds()throws {
        let directory=try fixture(),id=UUID().uuidString
        let owner=try WorkflowGroupOwnership.initialize(directory:directory,groupID:id)
        XCTAssertEqual(owner.metadata.groupID,id)
        XCTAssertEqual(owner.metadata.schemaVersion,1)
        XCTAssertNotNil(UUID(uuidString:owner.metadata.generation))
        XCTAssertNoThrow(try owner.check())
        XCTAssertEqual(try WorkflowGroupOwnership.readMetadata(directory:directory,groupID:id),owner.metadata)
        XCTAssertThrowsError(try WorkflowGroupOwnership.acquireRecovery(directory:directory,groupID:id))
        let activity=try WorkflowFileLease(directory:directory,name:"activity.lock",mode:.shared,expectedIdentity:owner.metadata.activityLockIdentity)
        XCTAssertNoThrow(try owner.check(),"Worker activity may coexist with its owner.")
        owner.close();owner.close()
        XCTAssertThrowsError(try owner.check())
        XCTAssertThrowsError(try WorkflowGroupOwnership.acquireRecovery(directory:directory,groupID:id))
        activity.close()
        let recovery=try WorkflowGroupOwnership.acquireRecovery(directory:directory,groupID:id)
        XCTAssertEqual(recovery.metadata,owner.metadata)
        XCTAssertNoThrow(try recovery.check())
        XCTAssertThrowsError(try WorkflowFileLease(directory:directory,name:"activity.lock",mode:.shared))
        XCTAssertThrowsError(try WorkflowGroupOwnership.acquireRecovery(directory:directory,groupID:id))
        recovery.close();recovery.close()
        XCTAssertThrowsError(try recovery.check())
        XCTAssertNoThrow(try WorkflowGroupOwnership.acquireRecovery(directory:directory,groupID:id))
    }
    func testLegacyAndDuplicateInitializationNeverInventOwnership()throws {
        let directory=try fixture(),id=UUID().uuidString
        XCTAssertThrowsError(try WorkflowGroupOwnership.readMetadata(directory:directory,groupID:id))
        XCTAssertThrowsError(try WorkflowGroupOwnership.acquireRecovery(directory:directory,groupID:id))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:directory.path),[])
        try Data("{}".utf8).write(to:directory.appendingPathComponent("group.json"))
        XCTAssertThrowsError(try WorkflowGroupOwnership.initialize(directory:directory,groupID:id))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:directory.path),["group.json"])
        try FileManager.default.removeItem(at:directory.appendingPathComponent("group.json"))
        let owner=try WorkflowGroupOwnership.initialize(directory:directory,groupID:id)
        let before=try Data(contentsOf:directory.appendingPathComponent("ownership.json"))
        XCTAssertThrowsError(try WorkflowGroupOwnership.initialize(directory:directory,groupID:id))
        owner.close()
        XCTAssertThrowsError(try WorkflowGroupOwnership.initialize(directory:directory,groupID:id))
        XCTAssertEqual(try Data(contentsOf:directory.appendingPathComponent("ownership.json")),before)
    }
    func testMetadataTamperingAndWrongGroupRejectCapabilities()throws {
        let directory=try fixture(),id="AAAAAAAA-BBBB-4CCC-8DDD-EEEEEEEEEEEE"
        let owner=try WorkflowGroupOwnership.initialize(directory:directory,groupID:id)
        XCTAssertThrowsError(try WorkflowGroupOwnership.readMetadata(directory:directory,groupID:UUID().uuidString))
        XCTAssertThrowsError(try WorkflowGroupOwnership.readMetadata(directory:directory,groupID:id.lowercased()))
        let path=directory.appendingPathComponent("ownership.json"),original=try Data(contentsOf:path)
        var altered=try XCTUnwrap(JSONSerialization.jsonObject(with:original) as? [String:Any])
        altered["generation"]=UUID().uuidString
        try JSONSerialization.data(withJSONObject:altered,options:[.sortedKeys]).write(to:path)
        XCTAssertThrowsError(try owner.check())
        owner.close()
        // A valid-looking new generation cannot establish metadata equality for a held capability.
        try original.write(to:path)
        let recovery=try WorkflowGroupOwnership.acquireRecovery(directory:directory,groupID:id)
        try Data("{}".utf8).write(to:path)
        XCTAssertThrowsError(try recovery.check())
        recovery.close()
        XCTAssertThrowsError(try WorkflowGroupOwnership.acquireRecovery(directory:directory,groupID:id))
    }
    func testReplacedActivityLockAndDirectoryAreRejected()throws {
        let directory=try fixture(),id=UUID().uuidString,fm=FileManager.default
        let owner=try WorkflowGroupOwnership.initialize(directory:directory,groupID:id)
        let activity=directory.appendingPathComponent("activity.lock")
        try fm.moveItem(at:activity,to:directory.appendingPathComponent("old-activity"))
        try Data().write(to:activity)
        XCTAssertThrowsError(try owner.check())
        owner.close()
        XCTAssertThrowsError(try WorkflowGroupOwnership.acquireRecovery(directory:directory,groupID:id))
        let replacement=try fixture()
        for name in ["ownership.json","owner.lock","activity.lock"] {try fm.copyItem(at:directory.appendingPathComponent(name),to:replacement.appendingPathComponent(name))}
        XCTAssertThrowsError(try WorkflowGroupOwnership.acquireRecovery(directory:replacement,groupID:id))
    }
    func testIncompleteLocksInvalidIdentifiersAndUnsafeMetadataStayUntouched()throws {
        let directory=try fixture(),fm=FileManager.default,id=UUID().uuidString
        XCTAssertThrowsError(try WorkflowGroupOwnership.initialize(directory:directory,groupID:"not-a-uuid"))
        XCTAssertEqual(try fm.contentsOfDirectory(atPath:directory.path),[])
        try Data().write(to:directory.appendingPathComponent("activity.lock"))
        XCTAssertThrowsError(try WorkflowGroupOwnership.initialize(directory:directory,groupID:id))
        XCTAssertEqual(try fm.contentsOfDirectory(atPath:directory.path),["activity.lock"])
        try fm.removeItem(at:directory.appendingPathComponent("activity.lock"))
        let owner=try WorkflowGroupOwnership.initialize(directory:directory,groupID:id)
        owner.close()
        let metadata=directory.appendingPathComponent("ownership.json"),backup=directory.appendingPathComponent("backup")
        try fm.moveItem(at:metadata,to:backup)
        try fm.createSymbolicLink(at:metadata,withDestinationURL:backup)
        XCTAssertThrowsError(try WorkflowGroupOwnership.acquireRecovery(directory:directory,groupID:id))
        try fm.removeItem(at:metadata)
        try Data(repeating:32,count:16385).write(to:metadata)
        XCTAssertThrowsError(try WorkflowGroupOwnership.readMetadata(directory:directory,groupID:id))
    }
}
