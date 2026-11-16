import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

final class ComponentFamilyStyleTests: XCTestCase {
    @MainActor func testActivationFamilyHasOneSharedTintAcrossPaletteCanvasAndInspector() {
        let baseline = NSColor(ComponentStyle.tint("relu")).usingColorSpace(.deviceRGB)!
        for type in ["tanh", "gelu", "softmax"] {
            let color = NSColor(ComponentStyle.tint(type)).usingColorSpace(.deviceRGB)!
            XCTAssertEqual(color.redComponent, baseline.redComponent, accuracy: 0.001, type)
            XCTAssertEqual(color.greenComponent, baseline.greenComponent, accuracy: 0.001, type)
            XCTAssertEqual(color.blueComponent, baseline.blueComponent, accuracy: 0.001, type)
        }
    }
    func testVisibleModelInputsUseTheCurrentBoundaryFamily() {
        for type in ["input","index_input","output"] {
            XCTAssertEqual(ComponentVisualFamily.resolve(type:type),.boundaries)
            XCTAssertEqual(ComponentVisualFamily.resolve(type:type).silhouette,.square)
        }
        XCTAssertEqual(ComponentArtworkCatalog.assets["input"],"model-input-v3")
        XCTAssertEqual(ComponentArtworkCatalog.assets["index_input"],"model-input-v3")
    }
    func testRegistryCategoriesRemainTheSourceOfVisualIdentity() throws {
        let registry = try ComponentRegistry.builtIn()
        for component in registry.components where !WorkflowModelInput.componentIDs.contains(component.id) && component.id != "separable_conv2d" && component.id != "dropout" {
            XCTAssertEqual(ComponentVisualFamily.resolve(type: component.id), ComponentVisualFamily.category(component.category), component.id)
        }
        XCTAssertEqual(ComponentVisualFamily.resolve(type: "kwip.data.csv"), .data)
        XCTAssertEqual(ComponentVisualFamily.resolve(type: "composite:custom"), .blocks)
        XCTAssertEqual(ComponentVisualFamily.resolve(type: "plugin.unrecognized"), .unknown)
    }
}
