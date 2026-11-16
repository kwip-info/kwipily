import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LinearFamilyAuthoringTests:XCTestCase {
    func testPaletteOffersOneActualDefaultAndRespectsAvailableSubset()throws {
        let registry=try ComponentRegistry.builtIn()
        func entries(_ ids:Set<String>)->[ComponentRibbonEntry] {registry.components.filter {ids.contains($0.id)}.map {.init(id:$0.id,title:$0.title,shortTitle:$0.title,purpose:"",category:$0.category,typeIcon:$0.id)}}
        let values=ComponentRibbonCatalog.visible(entries(LinearConfiguration.componentIDs.union(["bilinear"])),scope:.all,query:"")
        let linear=try XCTUnwrap(values.first {$0.typeIcon=="kwip.linear"})
        XCTAssertEqual(linear.id,"linear");XCTAssertEqual(linear.title,"Linear")
        XCTAssertEqual(ComponentRibbonCatalog.visible(entries(["linear"]),scope:.all,query:"Linear").count,1)
        XCTAssertEqual(values.filter {$0.typeIcon=="kwip.linear"}.count,1)
        XCTAssertFalse(values.contains {LinearConfiguration.componentIDs.contains($0.typeIcon)})
        let restricted=ComponentRibbonCatalog.visible(entries(["lazy_linear"]),scope:.all,query:"")
        XCTAssertEqual(restricted.map(\.id),["lazy_linear"])
        XCTAssertEqual(ComponentRibbonCatalog.visible(entries(["lazy_linear"]),scope:.all,query:"LazyLinear").map(\.id),["lazy_linear"])
        XCTAssertEqual(restricted.map(\.typeIcon),["kwip.linear"])
        XCTAssertTrue(ComponentRibbonCatalog.visible(entries(["lazy_linear"]),scope:.all,query:"Bilinear").isEmpty)
    }
    @MainActor func testWorkflowPaletteOpeningPreservesOldLinearSource()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        let address=WorkflowNodeAddress(domain:.model,localID:"normalizer")
        let original=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,name:"Existing linear",payload:.modelPrimitive(type:"lazy_linear",configurationVersion:2,config:try LinearConfiguration.defaults(for:.inferred,registry:.builtIn())))],edges:[])
        _=try library.saveSource(.workflow(original))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:original.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        defer {canvas.invalidate();owner.invalidate()}
        XCTAssertEqual(owner.document,original)
        XCTAssertEqual(canvas.creationEntries.filter {$0.component.title=="Linear"}.count,1)
        XCTAssertFalse(canvas.creationEntries.contains {LinearConfiguration.componentIDs.contains($0.iconType)})
    }
}
