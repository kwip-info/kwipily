import XCTest
import KwipMLCore
@testable import KwipMLApp

final class StatefulNormalizationPresentationTests: XCTestCase {
    @MainActor func testConsolidatedArtworkRetainsOriginalAssetsAndFamiliarLessonSearch() throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        for id in StatefulNormalizationConfiguration.componentIDs {
            XCTAssertNotNil(ComponentArtworkCatalog.image(for: id), id)
            XCTAssertEqual(ComponentArtworkCatalog.assetName(for: id), "normalization-v3")
            XCTAssertTrue(FileManager.default.fileExists(atPath:root.appendingPathComponent("Sources/KwipMLApp/Resources/ComponentIcons/"+id+"-v2.png").path),"Original artwork remains retained")
            XCTAssertEqual(ComponentVisualFamily.resolve(type: id), .normalization)
            XCTAssertFalse(ComponentStyle.purpose(id).isEmpty)
        }
        let lab = try XCTUnwrap(LearnCatalogContent.items.first { $0.id == "channel-normalization" })
        for keyword in ["BatchNorm", "InstanceNorm", "GroupNorm"] {
            XCTAssertTrue(lab.matches(section: "Library", search: keyword), keyword)
        }
    }
}
