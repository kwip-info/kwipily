import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ActivationAuthoringTests:XCTestCase {
    func testRestrictedPaletteConsolidatesOnlyAvailableFunctionsAndKeepsDropout()throws {
        let registry=try ComponentRegistry.builtIn()
        let entries=registry.components.filter {["sigmoid","linear","dropout"].contains($0.id)}.map {
            ComponentRibbonEntry(id:$0.id,title:$0.title,shortTitle:$0.title,purpose:"",category:$0.category,typeIcon:$0.id)
        }
        let values=ComponentRibbonCatalog.visible(entries,scope:.all,query:"")
        let activation=try XCTUnwrap(values.first {$0.typeIcon==ActivationPalette.type})
        XCTAssertEqual(activation.id,"sigmoid")
        XCTAssertEqual(activation.title,"Activation")
        XCTAssertEqual(values.filter {$0.typeIcon==ActivationPalette.type}.count,1)
        XCTAssertTrue(values.contains {$0.typeIcon=="dropout"})
        XCTAssertFalse(values.contains {ActivationConfiguration.componentIDs.contains($0.typeIcon)})
        XCTAssertEqual(ActivationPalette.functions(allowedTypes:["sigmoid"]),[.sigmoid])
        XCTAssertTrue(ComponentRibbonCatalog.visible(entries,scope:.all,query:"GELU").isEmpty)
    }
    @MainActor func testWorkflowPaletteAndInlineSwitchKeepOriginalProjectAndRenewContext()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root)
        let document=try WorkflowTrainedModelSource.promoteAuthoring(WorkflowExamples.preparedParallelClassifier(device:"cpu"))
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        defer {canvas.invalidate();owner.invalidate()}
        XCTAssertEqual(owner.document,document,"Opening the consolidated editor must not migrate scientific source")
        XCTAssertEqual(canvas.creationEntries.filter {$0.component.title=="Activation"}.count,1)
        XCTAssertFalse(canvas.creationEntries.contains {ActivationConfiguration.componentIDs.contains($0.iconType)})
        XCTAssertTrue(canvas.creationEntries.contains {$0.iconType=="dropout"})
        let address=WorkflowInlineContainer.address(root:.init(domain:.model,localID:"features"),childID:"activation-a")
        let display=try XCTUnwrap(canvas.projection.displayIDByAddress[address]);canvas.selectNode(display,extending:false)
        let context=try WorkflowModelNodeContext(owner:owner,address:address)
        let old=context.configuration
        XCTAssertTrue(context.setActivation(.gelu,configuration:["approximate_tanh":true],expectedType:"relu",expected:old),context.statusMessage)
        XCTAssertTrue(context.isCurrent)
        XCTAssertEqual(context.primitiveComponentID,"gelu")
        XCTAssertEqual(context.configuration,["approximate_tanh":true])
        XCTAssertEqual(owner.selection.nodeAddresses,[address])
        XCTAssertEqual(owner.document.edges,document.edges)
        XCTAssertTrue(owner.document.definitions.contains {document.definitions.contains($0)})
        XCTAssertFalse(context.setActivation(.tanh,configuration:[:],expectedType:"relu",expected:old),"Captured callback cannot replace a newer function")
        canvas.undo();XCTAssertEqual(owner.authoringNode(at:address)?.payload,.modelPrimitive(type:"relu",configurationVersion:2,config:[:]))
        XCTAssertFalse(context.isCurrent,"Undo must retire old inspector callbacks")
    }
}
