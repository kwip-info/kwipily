import Foundation
import XCTest
import KwipMLDatabaseNetworking

#if DEBUG
final class DatabaseSourceDevelopmentBookmarkTests: XCTestCase {
    func testEphemeralStaleMetadataCannotRedirectBoundCodeRoot() {
        let expected = URL(fileURLWithPath: "/tmp/KWIPBoundCode", isDirectory: true)
        for stale in [false, true] {
            XCTAssertTrue(DatabaseSourceServiceIdentity.matchesDevelopmentRoot(expected, expected: expected, bookmarkIsStale: stale))
            XCTAssertFalse(DatabaseSourceServiceIdentity.matchesDevelopmentRoot(URL(fileURLWithPath: "/tmp/OtherCode"), expected: expected, bookmarkIsStale: stale))
            XCTAssertFalse(DatabaseSourceServiceIdentity.matchesDevelopmentRoot(URL(fileURLWithPath: "/tmp/KWIPBoundCode/child"), expected: expected, bookmarkIsStale: stale))
        }
    }
}
#endif
