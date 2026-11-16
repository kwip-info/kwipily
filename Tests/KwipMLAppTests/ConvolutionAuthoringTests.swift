import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ConvolutionAuthoringTests:XCTestCase {
    func testPaletteOffersOneActualDefaultAndRespectsAvailableSubset()throws {
        let registry=try ComponentRegistry.builtIn()
        func entries(_ ids:Set<String>)->[ComponentRibbonEntry] {registry.components.filter {ids.contains($0.id)}.map {.init(id:$0.id,title:$0.title,shortTitle:$0.title,purpose:"",category:$0.category,typeIcon:$0.id)}}
        let values=ComponentRibbonCatalog.visible(entries(ConvolutionConfiguration.componentIDs.union(["linear"])),scope:.all,query:"")
        let convolution=try XCTUnwrap(values.first {$0.typeIcon=="kwip.convolution"})
        XCTAssertEqual(convolution.id,"conv2d");XCTAssertEqual(convolution.title,"Convolution")
        XCTAssertEqual(values.filter {$0.typeIcon=="kwip.convolution"}.count,1)
        XCTAssertFalse(values.contains {ConvolutionConfiguration.componentIDs.contains($0.typeIcon)})
        let restricted=ComponentRibbonCatalog.visible(entries(["conv_transpose1d"]),scope:.all,query:"")
        XCTAssertEqual(restricted.map(\.id),["conv_transpose1d"])
        XCTAssertEqual(ComponentRibbonCatalog.visible(entries(["conv_transpose1d"]),scope:.all,query:"ConvTranspose1d").map(\.id),["conv_transpose1d"])
        XCTAssertEqual(restricted.map(\.typeIcon),["kwip.convolution"])
        XCTAssertTrue(ComponentRibbonCatalog.visible(entries(["conv_transpose1d"]),scope:.all,query:"Depthwise").isEmpty)
    }
    @MainActor func testWorkflowPaletteOpeningPreservesOldConvolutionSource()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        let address=WorkflowNodeAddress(domain:.model,localID:"pool")
        let original=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,name:"Existing pool",payload:.modelPrimitive(type:"conv2d",configurationVersion:2,config:try ConvolutionConfiguration.defaults(for:.conv2d,registry:.builtIn())))],edges:[])
        _=try library.saveSource(.workflow(original))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:original.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        defer {canvas.invalidate();owner.invalidate()}
        XCTAssertEqual(owner.document,original)
        let firstLayers=try XCTUnwrap(canvas.creationEntries.firstIndex {$0.component.category=="Layers"})
        let firstPooling=try XCTUnwrap(canvas.creationEntries.firstIndex {$0.component.category=="Pooling"})
        XCTAssertLessThan(firstLayers,firstPooling,"Consolidation preserves the original Layers category position")
        XCTAssertEqual(canvas.creationEntries.filter {$0.component.title=="Convolution"}.count,1)
        XCTAssertFalse(canvas.creationEntries.contains {ConvolutionConfiguration.componentIDs.contains($0.iconType)})
    }
}
