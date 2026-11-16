import XCTest
@testable import KwipMLApp

final class SourcePermissionInventoryTests:XCTestCase {
    @MainActor func testWorkspaceRejectsBusyRevocationAndLegacyWorkerPathHonorsDenial() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let workspace=WorkspaceStore(rootOverride:root,storageOverride:root.appendingPathComponent("storage"),detectCapabilities:false)
        let file=root.appendingPathComponent("data.csv")
        try Data("a\n1\n".utf8).write(to:file)
        try workspace.sourceAccess.remember(file)
        let entry=try XCTUnwrap(workspace.sourceAccess.listPermissions().entries.first)
        workspace.isPreparingData=true
        XCTAssertThrowsError(try workspace.revokeSourcePermission(entry))
        XCTAssertEqual(try workspace.sourceAccess.listPermissions().entries.first?.state,.saved)
        workspace.isPreparingData=false
        try workspace.revokeSourcePermission(entry)
        XCTAssertThrowsError(try workspace.workerSourcePath(file.path))
        XCTAssertEqual(try workspace.sourceAccess.listPermissions().entries.first?.state,.revoked)
    }
    func testListingRevocationAndStaleSelectionPreserveOtherRecords() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let file=root.appendingPathComponent("data.csv")
        try Data("a\n1\n".utf8).write(to:file)
        let access=SourceAccess(directory:root.appendingPathComponent("permissions"))
        try access.remember(file)
        let entry=try XCTUnwrap(access.listPermissions().entries.first)
        XCTAssertEqual(entry.path,file.path,entry.detail ?? "");XCTAssertEqual(entry.state,.saved)
        let record=access.directory.appendingPathComponent(entry.id)
        let original=try Data(contentsOf:record)
        try (original+Data([0])).write(to:record)
        XCTAssertThrowsError(try access.revoke(entry:entry))
        XCTAssertEqual(try Data(contentsOf:record),original+Data([0]))
        try original.write(to:record)
        try access.revoke(entry:entry)
        let revoked=try XCTUnwrap(access.listPermissions().entries.first)
        XCTAssertEqual(revoked.path,file.path);XCTAssertEqual(revoked.state,.revoked)
        XCTAssertEqual(try Data(contentsOf:file),Data("a\n1\n".utf8))
        XCTAssertThrowsError(try access.withAccess(to:file.path) {_ in XCTFail("Revoked read")})
    }
    func testUnreadableAndSymlinkRecordsAreReportedAndScanIsBounded() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let access=SourceAccess(directory:root)
        let corrupt=root.appendingPathComponent(String(repeating:"a",count:64)+".bookmark")
        try Data("corrupt".utf8).write(to:corrupt)
        let link=root.appendingPathComponent(String(repeating:"b",count:64)+".bookmark")
        try FileManager.default.createSymbolicLink(at:link,withDestinationURL:corrupt)
        let listing=try access.listPermissions()
        XCTAssertEqual(listing.entries.count,1);XCTAssertEqual(listing.entries.first?.state,.unavailable)
        XCTAssertEqual(listing.issues.count,1)
        XCTAssertTrue(try access.listPermissions(limit:1).truncated)
        XCTAssertThrowsError(try access.listPermissions(limit:0))
    }
}
