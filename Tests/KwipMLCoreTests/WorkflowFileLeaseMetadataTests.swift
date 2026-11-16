import XCTest
import Foundation
import Darwin
@testable import KwipMLCore

final class WorkflowFileLeaseMetadataTests:XCTestCase {
    func testImmutableMetadataIsBoundedAndDoesNotFollowReplacedDirectory()throws {
        guard let resolved=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw NSError(domain:NSPOSIXErrorDomain,code:Int(errno))}
        defer {free(resolved)}
        let root=URL(fileURLWithPath:String(cString:resolved)).appendingPathComponent(UUID().uuidString)
        let directory=root.appendingPathComponent("group")
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let lease=try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.exclusive,create:true,createExclusive:true)
        XCTAssertThrowsError(try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.shared,create:true,createExclusive:true))
        XCTAssertNil(try lease.readOwnershipMetadata())
        XCTAssertFalse(try lease.containsSibling("group.json"))
        XCTAssertThrowsError(try lease.containsSibling("../outside"))
        XCTAssertThrowsError(try lease.publishOwnershipMetadata(Data(repeating:0,count:16385)))
        XCTAssertNil(try lease.readOwnershipMetadata())
        let original=Data("{\"schemaVersion\":1}".utf8)
        try lease.publishOwnershipMetadata(original)
        XCTAssertEqual(try lease.readOwnershipMetadata(),original)
        XCTAssertThrowsError(try lease.publishOwnershipMetadata(Data("replacement".utf8)))
        XCTAssertEqual(try lease.readOwnershipMetadata(),original)
        XCTAssertEqual(Set(try FileManager.default.contentsOfDirectory(atPath:directory.path)),["owner.lock","ownership.json"])
        try FileManager.default.moveItem(at:directory,to:root.appendingPathComponent("old"))
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
        XCTAssertThrowsError(try lease.publishOwnershipMetadata(original))
        XCTAssertThrowsError(try lease.readOwnershipMetadata())
        XCTAssertFalse(FileManager.default.fileExists(atPath:directory.appendingPathComponent("ownership.json").path))
    }
    func testAnchoredDirectoryAndFileSystemMetadataExpireWithLease()throws {
        guard let resolved=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw NSError(domain:NSPOSIXErrorDomain,code:Int(errno))}
        defer {free(resolved)}
        let directory=URL(fileURLWithPath:String(cString:resolved)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
        defer {try? FileManager.default.removeItem(at:directory)}
        let lease=try WorkflowFileLease(directory:directory,name:"owner.lock",mode:.exclusive,create:true)
        var directoryStat=stat();XCTAssertEqual(lstat(directory.path,&directoryStat),0)
        XCTAssertEqual(lease.parentIdentity,.init(device:UInt64(UInt32(bitPattern:directoryStat.st_dev)),inode:UInt64(directoryStat.st_ino)))
        let descriptor=open(directory.path,O_RDONLY|O_DIRECTORY|O_CLOEXEC)
        XCTAssertGreaterThanOrEqual(descriptor,0);defer {Darwin.close(descriptor)}
        var mount=statfs();XCTAssertEqual(fstatfs(descriptor,&mount),0)
        let name=withUnsafeBytes(of:mount.f_fstypename) {String(decoding:$0.prefix(while:{$0 != 0}),as:UTF8.self)}
        let info=try lease.fileSystemInfo()
        XCTAssertEqual(info.type,name);XCTAssertEqual(info.isLocal,mount.f_flags & UInt32(MNT_LOCAL) != 0)
        lease.close()
        XCTAssertThrowsError(try lease.fileSystemInfo()) {XCTAssertEqual($0 as? WorkflowFileLeaseError,.closed)}
    }
}
