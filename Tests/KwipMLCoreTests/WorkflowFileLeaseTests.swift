import XCTest
import Foundation
import Darwin
@testable import KwipMLCore

final class WorkflowFileLeaseTests:XCTestCase {
    private func fixture()throws->URL {
        guard let canonical=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw NSError(domain:NSPOSIXErrorDomain,code:Int(errno))}
        defer {free(canonical)}
        let base=URL(fileURLWithPath:String(cString:canonical),isDirectory:true).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:base,withIntermediateDirectories:false)
        addTeardownBlock {try? FileManager.default.removeItem(at:base)}
        return base
    }
    func testSeparateOpensContendAndExplicitCloseIsIdempotent()throws {
        let directory=try fixture()
        let owner=try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.exclusive,create:true)
        let identity=owner.identity
        XCTAssertThrowsError(try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.exclusive)) {XCTAssertEqual($0 as? WorkflowFileLeaseError,.busy)}
        XCTAssertThrowsError(try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.shared)) {XCTAssertEqual($0 as? WorkflowFileLeaseError,.busy)}
        XCTAssertNoThrow(try owner.checkIdentity())
        owner.close();owner.close()
        XCTAssertThrowsError(try owner.checkIdentity()) {XCTAssertEqual($0 as? WorkflowFileLeaseError,.closed)}
        let reopened=try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.exclusive,expectedIdentity:identity)
        XCTAssertEqual(reopened.identity,identity)
        reopened.close()
        XCTAssertEqual(try JSONDecoder().decode(WorkflowFileIdentity.self,from:JSONEncoder().encode(identity)),identity)
    }
    func testSharedCoexistenceDeinitAndMissingCreatePolicy()throws {
        let directory=try fixture()
        XCTAssertThrowsError(try WorkflowFileLease(directory:directory,name:"activity.lock",mode:.shared))
        var first:WorkflowFileLease?=try WorkflowFileLease(directory:directory,name:"activity.lock",mode:.shared,create:true)
        let second=try WorkflowFileLease(directory:directory,name:"activity.lock",mode:.shared,create:true)
        XCTAssertEqual(first?.identity,second.identity)
        XCTAssertThrowsError(try WorkflowFileLease(directory:directory,name:"activity.lock",mode:.exclusive))
        first=nil
        XCTAssertThrowsError(try WorkflowFileLease(directory:directory,name:"activity.lock",mode:.exclusive))
        second.close()
        XCTAssertNoThrow(try WorkflowFileLease(directory:directory,name:"activity.lock",mode:.exclusive))
    }
    func testRejectsNamesSymlinkLeafAndEverySymlinkDirectory()throws {
        let directory=try fixture(),fm=FileManager.default
        for name in ["", ".", "..", "other.lock", "../owner.lock", "nested/owner.lock", "owner.lock\0suffix"] {
            XCTAssertThrowsError(try WorkflowFileLease(directory:directory,name:name,mode:.exclusive,create:true)) {XCTAssertEqual($0 as? WorkflowFileLeaseError,.unsafePath)}
        }
        let target=directory.appendingPathComponent("target")
        try Data().write(to:target)
        try fm.createSymbolicLink(at:directory.appendingPathComponent("owner.lock"),withDestinationURL:target)
        XCTAssertThrowsError(try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.exclusive,create:true))
        let real=directory.appendingPathComponent("real"),link=directory.appendingPathComponent("link")
        try fm.createDirectory(at:real.appendingPathComponent("child"),withIntermediateDirectories:true)
        try fm.createSymbolicLink(at:link,withDestinationURL:real)
        for path in [link,link.appendingPathComponent("child")] {
            XCTAssertThrowsError(try WorkflowFileLease(directory:path,name:"activity.lock",mode:.shared,create:true))
        }
        XCTAssertFalse(fm.fileExists(atPath:real.appendingPathComponent("activity.lock").path))
    }
    func testHardlinksReplacementAndExpectedIdentityMismatch()throws {
        let directory=try fixture(),fm=FileManager.default
        let lease=try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.exclusive,create:true)
        let original=lease.identity,path=directory.appendingPathComponent("owner.lock"),alias=directory.appendingPathComponent("alias")
        try fm.linkItem(at:path,to:alias)
        XCTAssertThrowsError(try lease.checkIdentity())
        XCTAssertThrowsError(try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.shared))
        try fm.removeItem(at:alias)
        XCTAssertNoThrow(try lease.checkIdentity())
        try fm.moveItem(at:path,to:directory.appendingPathComponent("old-lock"))
        try Data().write(to:path)
        XCTAssertThrowsError(try lease.checkIdentity())
        XCTAssertThrowsError(try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.exclusive,expectedIdentity:original))
        lease.close()
        let replacement=try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.exclusive)
        XCTAssertNotEqual(replacement.identity,original)
    }
    func testDirectoryReplacementInvalidatesAnchoredLease()throws {
        let parent=try fixture(),fm=FileManager.default,directory=parent.appendingPathComponent("run")
        try fm.createDirectory(at:directory,withIntermediateDirectories:false)
        let lease=try WorkflowFileLease(directory:directory,name:"activity.lock",mode:.shared,create:true)
        try fm.moveItem(at:directory,to:parent.appendingPathComponent("old"))
        try fm.createDirectory(at:directory,withIntermediateDirectories:false)
        try Data().write(to:directory.appendingPathComponent("activity.lock"))
        XCTAssertThrowsError(try lease.checkIdentity())
    }
    func testWritableDirectoryAndLockAreRejectedBeforeAndDuringLease()throws {
        let directory=try fixture(),path=directory.appendingPathComponent("owner.lock")
        let held=try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.exclusive,create:true)
        XCTAssertEqual(chmod(directory.path,0o777),0)
        XCTAssertThrowsError(try held.checkIdentity())
        XCTAssertThrowsError(try WorkflowFileLease(directory:directory,name:"activity.lock",mode:.shared,create:true))
        XCTAssertEqual(chmod(directory.path,0o700),0)
        XCTAssertNoThrow(try held.checkIdentity())
        XCTAssertEqual(chmod(path.path,0o666),0)
        XCTAssertThrowsError(try held.checkIdentity())
        held.close()
        XCTAssertThrowsError(try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.exclusive))
        XCTAssertEqual(chmod(path.path,0o600),0)
        XCTAssertNoThrow(try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.exclusive))
    }
    func testIndependentPythonProcessObservesKernelLeaseAndRelease()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python=root.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath:python.path) else {throw XCTSkip("Development Python unavailable for independent fcntl interoperability check.")}
        let directory=try fixture(),lease=try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.exclusive,create:true)
        func attempt()throws->Int32 {
            let process=Process();process.executableURL=python
            process.arguments=["-c","import fcntl, os, sys\nf=os.open(sys.argv[1],os.O_RDWR)\ntry:\n fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError:\n sys.exit(23)\nsys.exit(0)",directory.appendingPathComponent("owner.lock").path]
            process.standardOutput=FileHandle.nullDevice;process.standardError=FileHandle.nullDevice
            try process.run();process.waitUntilExit()
            XCTAssertEqual(process.terminationReason,.exit)
            return process.terminationStatus
        }
        XCTAssertEqual(try attempt(),23)
        lease.close()
        XCTAssertEqual(try attempt(),0)
    }
    func testLeaseDoesNotRequireReadPermissionOnSearchableAncestors()throws {
        guard geteuid() != 0 else {throw XCTSkip("Root bypasses ancestor read permission checks")}
        let ancestor=try fixture(),directory=ancestor.appendingPathComponent("allowed-child",isDirectory:true)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false,attributes:[.posixPermissions:0o700])
        XCTAssertEqual(chmod(ancestor.path,0o111),0)
        defer {XCTAssertEqual(chmod(ancestor.path,0o700),0)}
        let denied=open(ancestor.path,O_RDONLY|O_DIRECTORY)
        let code=errno
        if denied>=0 {Darwin.close(denied);return XCTFail("Fixture must deny ancestor directory reads")}
        XCTAssertEqual(code,EACCES)
        let lease=try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.exclusive,create:true)
        defer {lease.close()}
        try lease.checkIdentity()
        try lease.publishGroupRecord(Data("allowed final directory".utf8))
        XCTAssertEqual(try lease.readGroupRecord(),Data("allowed final directory".utf8))
    }

}
