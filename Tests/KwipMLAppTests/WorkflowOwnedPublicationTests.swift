import XCTest
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class WorkflowOwnedPublicationTests:XCTestCase {
    func testOwnerValidatesExactDirectoryAndRetainedAuthorityForPublication()throws {
        guard let canonical=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw NSError(domain:NSPOSIXErrorDomain,code:Int(errno))}
        defer {free(canonical)}
        let root=URL(fileURLWithPath:String(cString:canonical)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
        defer {try? FileManager.default.removeItem(at:root)}
        let owner=try WorkflowGroupOwnership.initialize(directory:root,groupID:root.lastPathComponent)
        defer {owner.close()}
        try owner.validateDirectory(root)
        XCTAssertThrowsError(try owner.validateDirectory(root.appendingPathComponent("other")))
        let first=Data("first".utf8),second=Data("second".utf8)
        try owner.publishGroupAuthoring(Data("authoring".utf8))
        try owner.publishGroupRecord(first)
        try owner.replaceGroupRecord(expected:first,data:second)
        XCTAssertEqual(try Data(contentsOf:root.appendingPathComponent("group.json")),second)
        XCTAssertThrowsError(try owner.replaceGroupRecord(expected:first,data:Data()))
        owner.close()
        XCTAssertThrowsError(try owner.validateDirectory(root))
        XCTAssertThrowsError(try owner.replaceGroupRecord(expected:second,data:Data()))
        XCTAssertEqual(try Data(contentsOf:root.appendingPathComponent("group.json")),second)
    }
    func testCreateGroupCanonicalizesWorkspaceAndRetainsExclusiveOwner()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
        defer {try? FileManager.default.removeItem(at:root)}
        let project=UUID().uuidString
        let result=try WorkflowGroupOwnership.createGroup(workspace:root,projectID:project)
        defer {result.owner.close()}
        guard let canonical=realpath(result.directory.path,nil) else {return XCTFail("Missing group directory")}
        defer {free(canonical)}
        XCTAssertEqual(result.directory.path,String(cString:canonical))
        XCTAssertEqual(result.directory.deletingLastPathComponent().lastPathComponent,project)
        XCTAssertEqual(result.owner.metadata.groupID,result.directory.lastPathComponent)
        try result.owner.validateDirectory(result.directory)
        XCTAssertThrowsError(try WorkflowGroupOwnership.acquireRecovery(directory:result.directory,groupID:result.directory.lastPathComponent))
        result.owner.close()
        let recovered=try WorkflowGroupOwnership.acquireRecovery(directory:result.directory,groupID:result.directory.lastPathComponent)
        recovered.close()
    }
    func testCreateGroupRejectsRedirectedExperimentAndProjectDirectoriesBeforeWrites()throws {
        for redirectProject in [false,true] {
            let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
            defer {try? FileManager.default.removeItem(at:root)}
            let destination=root.appendingPathComponent("unrelated")
            try FileManager.default.createDirectory(at:destination,withIntermediateDirectories:false)
            try Data("untouched".utf8).write(to:destination.appendingPathComponent("sentinel"))
            let project=UUID().uuidString,experiments=root.appendingPathComponent("workflow-experiments")
            if redirectProject {try FileManager.default.createDirectory(at:experiments,withIntermediateDirectories:false)}
            try FileManager.default.createSymbolicLink(at:redirectProject ? experiments.appendingPathComponent(project):experiments,withDestinationURL:destination)
            XCTAssertThrowsError(try WorkflowGroupOwnership.createGroup(workspace:root,projectID:project))
            XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:destination.path),["sentinel"])
            XCTAssertEqual(try Data(contentsOf:destination.appendingPathComponent("sentinel")),Data("untouched".utf8))
        }
    }

}
