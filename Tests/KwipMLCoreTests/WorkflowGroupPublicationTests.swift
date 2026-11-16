import XCTest
import Darwin
@testable import KwipMLCore

final class WorkflowGroupPublicationTests:XCTestCase {
    private func folder()throws->URL {
        guard let canonical=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw NSError(domain:NSPOSIXErrorDomain,code:Int(errno))}
        defer {free(canonical)}
        let root=URL(fileURLWithPath:String(cString:canonical)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        return root
    }
    func testExclusiveOwnerPublishesAndConditionallyReplacesCompleteBytes()throws {
        let root=try folder(),lease=try WorkflowFileLease(directory:root,name:"owner.lock",mode:.exclusive,create:true)
        defer {lease.close()}
        let first=Data(repeating:65,count:255999),second=Data(repeating:66,count:256000)
        try lease.publishGroupRecord(first)
        XCTAssertEqual(try lease.readGroupRecord(),first)
        XCTAssertThrowsError(try lease.publishGroupRecord(second))
        XCTAssertThrowsError(try lease.replaceGroupRecord(expected:Data("stale".utf8),data:second))
        XCTAssertEqual(try lease.readGroupRecord(),first)
        try lease.replaceGroupRecord(expected:first,data:second)
        XCTAssertEqual(try lease.readGroupRecord(),second)
        XCTAssertThrowsError(try lease.replaceGroupRecord(expected:second,data:Data(repeating:0,count:256001)))
        try lease.publishGroupAuthoring(Data(repeating:7,count:2_000_000))
        XCTAssertEqual(try Data(contentsOf:root.appendingPathComponent("authoring.json")).count,2_000_000)
        XCTAssertThrowsError(try lease.publishGroupAuthoring(Data()))
        XCTAssertEqual(Set(try FileManager.default.contentsOfDirectory(atPath:root.path)),["owner.lock","group.json","authoring.json"])
    }
    func testSharedOrActivityLeaseCannotMutateGroupFiles()throws {
        for (name,mode) in [("owner.lock",WorkflowFileLease.Mode.shared),("activity.lock",.shared),("activity.lock",.exclusive)] {
            let root=try folder(),lease=try WorkflowFileLease(directory:root,name:name,mode:mode,create:true)
            defer {lease.close()}
            XCTAssertThrowsError(try lease.publishGroupRecord(Data()))
            XCTAssertThrowsError(try lease.replaceGroupRecord(expected:Data(),data:Data()))
            XCTAssertThrowsError(try lease.publishGroupAuthoring(Data()))
            XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:root.path),[name])
        }
    }
    func testUnsafeExistingTargetAndReplacedLockArePreserved()throws {
        for mutation in ["symlink","hardlink","writable","replacedLock","closed"] {
            let root=try folder(),lease=try WorkflowFileLease(directory:root,name:"owner.lock",mode:.exclusive,create:true)
            defer {lease.close()}
            let bytes=Data("original".utf8),path=root.appendingPathComponent("group.json")
            try lease.publishGroupRecord(bytes)
            if mutation=="symlink" {
                try FileManager.default.moveItem(at:path,to:root.appendingPathComponent("original"))
                try FileManager.default.createSymbolicLink(at:path,withDestinationURL:root.appendingPathComponent("original"))
            } else if mutation=="hardlink" {XCTAssertEqual(link(path.path,root.appendingPathComponent("alias").path),0)}
            else if mutation=="writable" {XCTAssertEqual(chmod(path.path,0o666),0)}
            else if mutation=="replacedLock" {
                try FileManager.default.moveItem(at:root.appendingPathComponent("owner.lock"),to:root.appendingPathComponent("old.lock"))
                try Data().write(to:root.appendingPathComponent("owner.lock"))
            } else {lease.close()}
            let before=Set(try FileManager.default.contentsOfDirectory(atPath:root.path))
            XCTAssertThrowsError(try lease.replaceGroupRecord(expected:bytes,data:Data("changed".utf8)))
            XCTAssertEqual(try Data(contentsOf:path),bytes)
            XCTAssertEqual(Set(try FileManager.default.contentsOfDirectory(atPath:root.path)),before)
        }
    }
}
