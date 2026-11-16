import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class CheckpointLifecycleArtworkTests: XCTestCase {
    func testLifecycleUsesOriginalGeneratedAssetWithoutAddingCanvasComponent() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let original = try Data(contentsOf: root.appendingPathComponent("docs/design/lifecycle-icons-v1/checkpoint-save-restore-v1.png"))
        let resource = try Data(contentsOf: root.appendingPathComponent("Sources/KwipMLApp/Resources/ComponentIcons/checkpoint-save-restore-v1.png"))
        XCTAssertEqual(original, resource)
        XCTAssertEqual(WorkflowArchiveCodec.hash(resource), "6fe639fbcae65fcd07ecec152855f731644725d281b93f1a01205abfeb2fbcc4")
        XCTAssertEqual(BundledLifecycleCapabilities.entry.presentation.icon, "docs/design/lifecycle-icons-v1/checkpoint-save-restore-v1.png")
        XCTAssertEqual(ComponentArtworkCatalog.checkpointLifecycleAssetName, "checkpoint-save-restore-v1")
        let image = try XCTUnwrap(ComponentArtworkCatalog.checkpointLifecycleImage)
        XCTAssertGreaterThanOrEqual(image.size.width, 48)
        XCTAssertGreaterThanOrEqual(image.size.height, 48)
        XCTAssertNil(try ComponentRegistry.builtIn().component(type: "checkpoint_lifecycle"))
        XCTAssertNil(ComponentArtworkCatalog.assetName(for: "checkpoint_lifecycle"))
    }
}
