import XCTest
import KwipMLCore
@testable import KwipMLApp

final class NormalizationAuthoringTests:XCTestCase {
    func testPaletteOffersOneActualDefaultAndRespectsAvailableSubset()throws {
        let registry=try ComponentRegistry.builtIn()
        func entries(_ ids:Set<String>)->[ComponentRibbonEntry] {registry.components.filter {ids.contains($0.id)}.map {.init(id:$0.id,title:$0.title,shortTitle:$0.title,purpose:"",category:$0.category,typeIcon:$0.id)}}
        let values=ComponentRibbonCatalog.visible(entries(NormalizationConfiguration.componentIDs.union(["linear"])),scope:.all,query:"")
        let normalization=try XCTUnwrap(values.first {$0.typeIcon=="kwip.normalization"})
        XCTAssertEqual(normalization.id,"layer_norm");XCTAssertEqual(normalization.title,"Normalization")
        XCTAssertEqual(ComponentRibbonCatalog.visible(entries(["layer_norm"]),scope:.all,query:"LayerNorm").count,1)
        XCTAssertEqual(values.filter {$0.typeIcon=="kwip.normalization"}.count,1)
        XCTAssertFalse(values.contains {NormalizationConfiguration.componentIDs.contains($0.typeIcon)})
        let restricted=ComponentRibbonCatalog.visible(entries(["group_norm"]),scope:.all,query:"")
        XCTAssertEqual(restricted.map(\.id),["group_norm"])
        XCTAssertEqual(ComponentRibbonCatalog.visible(entries(["group_norm"]),scope:.all,query:"GroupNorm").map(\.id),["group_norm"])
        XCTAssertEqual(restricted.map(\.typeIcon),["kwip.normalization"])
        XCTAssertTrue(ComponentRibbonCatalog.visible(entries(["group_norm"]),scope:.all,query:"BatchNorm").isEmpty)
    }
    @MainActor func testWorkflowPaletteOpeningPreservesOldNormalizationSource()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        let address=WorkflowNodeAddress(domain:.model,localID:"normalizer")
        let original=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,name:"Existing normalization",payload:.modelPrimitive(type:"layer_norm",configurationVersion:2,config:try NormalizationConfiguration.defaults(for:.layerNorm,registry:.builtIn())))],edges:[])
        _=try library.saveSource(.workflow(original))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:original.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        defer {canvas.invalidate();owner.invalidate()}
        XCTAssertEqual(owner.document,original)
        XCTAssertEqual(canvas.creationEntries.filter {$0.component.title=="Normalization"}.count,1)
        XCTAssertFalse(canvas.creationEntries.contains {NormalizationConfiguration.componentIDs.contains($0.iconType)})
    }
}
