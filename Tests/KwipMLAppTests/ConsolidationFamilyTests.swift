import XCTest
import KwipMLCore
@testable import KwipMLApp
@MainActor final class ConsolidationFamilyTests:XCTestCase {
    func testBoundaryAndContainerArtworkAgreeWithTheirGroups()throws {
        XCTAssertEqual(ComponentVisualFamily.resolve(type:"output"),.boundaries)
        XCTAssertEqual(ComponentVisualFamily.resolve(type:"workflow.modelDefinition"),.boundaries)
        XCTAssertEqual(ComponentVisualFamily.resolve(type:"workflow.saveModel"),.training)
        XCTAssertEqual(ComponentVisualFamily.resolve(type:"parallel_features"),.merge)
        XCTAssertEqual(ComponentArtworkCatalog.assetName(for:"output"),ComponentArtworkCatalog.assetName(for:"workflow.modelDefinition"))
        XCTAssertEqual(ComponentArtworkCatalog.assetName(for:"parallel_features"),"parallel-features-v1")
    }
    func testLayerOrderUsesCurrentToolsAndRetainsCustomEntries() {
        func entry(_ type:String)->ComponentRibbonEntry {.init(id:type,title:type,shortTitle:type,purpose:"",category:"Layers",typeIcon:type)}
        let entries=["kwip.convolution","owner.custom-layer","kwip.linear","bilinear","transformer"].map(entry)
        XCTAssertEqual(ComponentRibbonCatalog.visible(entries,scope:.category("Layers"),query:"").map(\.typeIcon),["kwip.linear","bilinear","kwip.convolution","transformer","owner.custom-layer"])
    }
}
