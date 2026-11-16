import XCTest
@testable import KwipMLApp

@MainActor final class Data02OutputDestinationCanonicalTests:XCTestCase {
    func testPOSIXCanonicalGrantKeepsPrivateVarSpellingAndRejectsUserLinks()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let folder=root.appendingPathComponent("saved-output")
        try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
        let canonical=try DataOutputDestinationProvider.canonicalGrantedPath(folder)
        XCTAssertTrue(canonical.hasPrefix("/private/var/"),"POSIX worker binding must not contain the macOS /var symlink alias: \(canonical)")
        XCTAssertTrue(canonical.hasSuffix("/saved-output"))
        let link=root.appendingPathComponent("user-link")
        try FileManager.default.createSymbolicLink(at:link,withDestinationURL:folder)
        XCTAssertThrowsError(try DataOutputDestinationProvider.canonicalGrantedPath(link))
    }
}
