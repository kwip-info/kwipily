import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ColumnsPresentationTests: XCTestCase {
    @MainActor func testNewColumnsHasAnExplicitIdentityRatherThanLegacyOrFallbackCopy() {
        XCTAssertEqual(ComponentStyle.shortTitle("kwip.data.columns", fallback: "Unknown"), "Columns")
        XCTAssertTrue(ComponentStyle.purpose("kwip.data.columns").contains("rename"))
        XCTAssertEqual(ComponentArtworkCatalog.assetName(for: "kwip.data.columns"), "data-columns-v1")
        XCTAssertEqual(ComponentStyle.shortTitle("kwip.data.selectRename", fallback: "Unknown"), "Select / name")
        XCTAssertEqual(ComponentStyle.shortTitle("kwip.data.cast", fallback: "Unknown"), "Cast")
    }
}
