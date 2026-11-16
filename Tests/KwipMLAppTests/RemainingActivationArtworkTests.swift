import AppKit
import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class RemainingActivationArtworkTests: XCTestCase {
    private let members: [(id: String, title: String, purpose: String)] = [
        ("prelu", "PReLU", "Learn how strongly negative measurements contribute."),
        ("log_softmax", "LogSoftmax", "Compare sample scores on a compressed logarithmic scale."),
        ("dropout", "Dropout", "Temporarily hide selected values during training, then keep every value for evaluation.")
    ]

    func testEveryRemainingActivationHasItsOwnDecodedOriginalArtwork() throws {
        for member in members {
            XCTAssertEqual(ComponentArtworkCatalog.assetName(for: member.id), member.id == "dropout" ? "dropout-v3":member.id + "-v2", member.id)
            let image = try XCTUnwrap(ComponentArtworkCatalog.image(for: member.id), member.id)
            let data = try XCTUnwrap(image.tiffRepresentation, member.id)
            let bitmap = try XCTUnwrap(NSBitmapImageRep(data: data), member.id)
            XCTAssertGreaterThan(bitmap.pixelsWide, 0, member.id)
            XCTAssertGreaterThan(bitmap.pixelsHigh, 0, member.id)
            XCTAssertTrue(bitmap.hasAlpha, "Original transparent icon: \(member.id)")
        }
    }

    func testInventoryRegistersDistinctAssetsWithReadableActivationTitles() throws {
        struct Catalog: Decodable {
            struct Entry: Decodable { let id: String; let title: String; let category: String; let asset: String }
            let components: [Entry]
        }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let catalog = try JSONDecoder().decode(Catalog.self, from: Data(contentsOf: root.appendingPathComponent("docs/design/component-icons-v2/catalog.json")))
        for member in members {
            let matches = catalog.components.filter { $0.id == member.id }
            XCTAssertEqual(matches.count, 1, member.id)
            let entry = try XCTUnwrap(matches.first, member.id)
            XCTAssertEqual(entry.title, member.title)
            XCTAssertEqual(entry.category, member.id == "dropout" ? "Conditioning":"Activations")
            XCTAssertEqual(entry.asset, member.id == "dropout" ? "dropout-v3.png":member.id + "-v2.png")
        }
    }

    func testPaletteAndCanvasShareReadableLabelsAndAmberActivationGrouping() throws {
        let registry = try ComponentRegistry.builtIn()
        for member in members {
            let definition = try XCTUnwrap(registry.components.first { $0.id == member.id })
            XCTAssertEqual(definition.title, member.title)
            XCTAssertEqual(definition.category, "Activations")
            XCTAssertEqual(ComponentStyle.shortTitle(member.id, fallback: "Unlabelled component"), member.title)
            XCTAssertEqual(ComponentStyle.purpose(member.id), member.purpose)
            let family = ComponentVisualFamily.resolve(type: member.id)
            XCTAssertEqual(family, member.id == "dropout" ? .normalization:.activations)
            XCTAssertEqual(family.colors.primary, member.id == "dropout" ? 0xB8BB8E:0xD7B57B)
            XCTAssertEqual(family.colors.secondary, member.id == "dropout" ? 0x757945:0x947443)
            XCTAssertEqual(family.silhouette, member.id == "dropout" ? .octagon:.circle)
        }
    }
}
