import XCTest
import AppKit
@testable import KwipMLApp

@MainActor final class SpatialPatchArtworkTests: XCTestCase {
    func testOriginalGeneratedAssetsAreAvailableAndDistinct() throws {
        let unfold = try XCTUnwrap(ComponentArtworkCatalog.image(for:"unfold"))
        let fold = try XCTUnwrap(ComponentArtworkCatalog.image(for:"fold"))
        XCTAssertNotEqual(unfold.tiffRepresentation,fold.tiffRepresentation)
        for image in [unfold,fold] {
            XCTAssertGreaterThan(image.size.width,0)
            XCTAssertGreaterThan(image.size.height,0)
        }
    }
}
