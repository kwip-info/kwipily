import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class EmbeddingBagArtworkTests:XCTestCase {
    func testPublishedPairUsesOriginalGeneratedArtworkInActualCatalog()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        for (type,name) in [("bag_input","bag-input-v1"),("embedding_bag","embedding-bag-v1")] {
            let original=try Data(contentsOf:root.appendingPathComponent("docs/design/component-icons-v2/\(name).png"))
            let resource=try Data(contentsOf:root.appendingPathComponent("Sources/KwipMLApp/Resources/ComponentIcons/\(name).png"))
            XCTAssertEqual(original,resource)
            XCTAssertEqual(ComponentArtworkCatalog.assetName(for:type),name)
            let image=try XCTUnwrap(ComponentArtworkCatalog.image(for:type))
            XCTAssertGreaterThanOrEqual(image.size.width,48)
            XCTAssertGreaterThanOrEqual(image.size.height,48)
        }
        XCTAssertEqual(ComponentArtworkCatalog.assetName(for:"embedding"),"embedding-v2")
    }
}
