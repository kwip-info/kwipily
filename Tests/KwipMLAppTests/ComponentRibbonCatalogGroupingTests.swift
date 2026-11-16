import XCTest
@testable import KwipMLApp

final class ComponentRibbonCatalogGroupingTests: XCTestCase {
    private func entry(_ id: String, type: String, category: String, purpose: String = "") -> ComponentRibbonEntry {
        .init(id: id, title: id, shortTitle: id, purpose: purpose, category: category, typeIcon: type)
    }
    func testCommonUsesAvailableTypesAndPreservesRealCreationIDs() {
        let values = [entry("creation-relu", type: "relu", category: "Activations"),
                      entry("creation-linear", type: "linear", category: "Layers"),
                      entry("creation-conv", type: "conv3d", category: "Layers"),
                      entry("creation-input", type: "input", category: "Data")]
        let visible = ComponentRibbonCatalog.visible(values, scope: .common, query: "")
        XCTAssertEqual(Set(visible.map(\.id)), ["creation-relu", "creation-linear", "creation-input"])
        XCTAssertFalse(visible.contains { $0.id == "linear" }, "Registry creation IDs must survive curation for add/drag")
        XCTAssertEqual(visible.map(\.typeIcon), ["input", LinearPalette.type, ActivationPalette.type], "Static Common priority is explicit, not personal telemetry")
    }
    func testSpecializedCatalogWithNoCommonMatchFallsBackToAvailableEntries() {
        let values = [entry("first", type: "custom.spatial", category: "Convolution"), entry("second", type: "custom.pool", category: "Pooling")]
        XCTAssertEqual(ComponentRibbonCatalog.visible(values, scope: .common, query: ""), values)
        XCTAssertTrue(ComponentRibbonCatalog.visible([], scope: .common, query: "").isEmpty)
    }
    func testGroupsFollowAvailableRegistryCategoryOrderWithoutMixingCategories() {
        let values = [entry("relu", type: "relu", category: "Activations"),
                      entry("linear", type: "linear", category: "Layers"),
                      entry("input", type: "input", category: "Data"),
                      entry("conv", type: "conv2d", category: "Layers")]
        let groups = ComponentRibbonCatalog.groups(values, scope: .all, query: "")
        XCTAssertEqual(groups.map(\.category), ["Data", "Layers", "Activations"])
        XCTAssertEqual(groups[1].entries.map(\.id), ["linear", "conv"])
        XCTAssertTrue(groups.allSatisfy { group in group.entries.allSatisfy { $0.category == group.category } })
        XCTAssertEqual(ComponentRibbonCatalog.groups(values, scope: .common, query: "").map(\.category), ["Data", "Layers", "Activations"])
    }
    func testSearchOverridesCommonAndCategoryAndRetainsAvailableGrouping() {
        let values = [entry("linear", type: "linear", category: "Layers"),
                      entry("rare", type: "custom.spatial", category: "Convolution", purpose: "Spatial training")]
        XCTAssertEqual(ComponentRibbonCatalog.visible(values, scope: .common, query: " TRAINING ").map(\.id), ["rare"])
        XCTAssertEqual(ComponentRibbonCatalog.groups(values, scope: .category("Layers"), query: "spatial").map(\.category), ["Convolution"])
        XCTAssertTrue(ComponentRibbonCatalog.groups(values, scope: .all, query: "does not exist").isEmpty)
        XCTAssertEqual(ComponentRibbonCatalog.visible(values, scope: .category("removed"), query: "").map(\.id), values.map(\.id))
    }
}
