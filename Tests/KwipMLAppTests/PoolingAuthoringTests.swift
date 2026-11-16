import XCTest
import KwipMLCore
@testable import KwipMLApp

final class PoolingAuthoringTests:XCTestCase {
    func testPaletteOffersOneActualDefaultAndRespectsAvailableSubset()throws {
        let registry=try ComponentRegistry.builtIn()
        func entries(_ ids:Set<String>)->[ComponentRibbonEntry] {registry.components.filter {ids.contains($0.id)}.map {.init(id:$0.id,title:$0.title,shortTitle:$0.title,purpose:"",category:$0.category,typeIcon:$0.id)}}
        let values=ComponentRibbonCatalog.visible(entries(PoolingConfiguration.componentIDs.union(["linear"])),scope:.all,query:"")
        let pooling=try XCTUnwrap(values.first {$0.typeIcon=="kwip.pooling"})
        XCTAssertEqual(pooling.id,"max_pool2d");XCTAssertEqual(pooling.title,"Pooling")
        XCTAssertEqual(values.filter {$0.typeIcon=="kwip.pooling"}.count,1)
        XCTAssertFalse(values.contains {PoolingConfiguration.componentIDs.contains($0.typeIcon)})
        let restricted=ComponentRibbonCatalog.visible(entries(["avg_pool1d"]),scope:.all,query:"")
        XCTAssertEqual(restricted.map(\.id),["avg_pool1d"])
        XCTAssertEqual(restricted.map(\.typeIcon),["kwip.pooling"])
        XCTAssertTrue(ComponentRibbonCatalog.visible(entries(["avg_pool1d"]),scope:.all,query:"Max").isEmpty)
    }
    @MainActor func testWorkflowPaletteOpeningPreservesOldPoolingSource()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        let address=WorkflowNodeAddress(domain:.model,localID:"pool")
        let original=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,name:"Existing pool",payload:.modelPrimitive(type:"max_pool2d",configurationVersion:2,config:try PoolingConfiguration.defaults(for:.maxPool2d,registry:.builtIn())))],edges:[])
        _=try library.saveSource(.workflow(original))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:original.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        defer {canvas.invalidate();owner.invalidate()}
        XCTAssertEqual(owner.document,original)
        XCTAssertEqual(canvas.creationEntries.filter {$0.component.title=="Pooling"}.count,1)
        XCTAssertFalse(canvas.creationEntries.contains {PoolingConfiguration.componentIDs.contains($0.iconType)})
    }
}
