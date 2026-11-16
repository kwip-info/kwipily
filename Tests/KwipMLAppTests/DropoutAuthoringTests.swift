import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class DropoutAuthoringTests:XCTestCase {
    func testOneConditioningEntryPreservesActualOperationAndRestrictedPalette()throws {
        let registry=try ComponentRegistry.builtIn()
        let component=try XCTUnwrap(registry.component(type:"dropout"))
        let entry=ComponentRibbonEntry(id:component.id,title:component.title,shortTitle:component.title,purpose:"Drop individual values",category:component.category,typeIcon:component.id)
        let offered=ComponentRibbonCatalog.visible([entry],scope:.category("Conditioning"),query:"Dropout")
        XCTAssertEqual(offered.map(\.id),["dropout"])
        let activation=ComponentRibbonEntry(id:"relu",title:"ReLU",shortTitle:"ReLU",purpose:"",category:"Activations",typeIcon:"relu")
        XCTAssertFalse(ComponentRibbonCatalog.visible([entry,activation],scope:.category("Activations"),query:"").contains {$0.typeIcon=="dropout"})
        XCTAssertEqual(ComponentVisualFamily.resolve(type:"dropout"),.normalization)
        XCTAssertEqual(ComponentArtworkCatalog.assetName(for:"dropout"),"dropout-v3")
    }
    func testOriginalSourceAtomicProbabilityUndoAndStaleEditRefusal()throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory)
        let address=WorkflowNodeAddress(domain:.model,localID:"dropout")
        let doc=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,name:"Original dropout",payload:.modelPrimitive(type:"dropout",configurationVersion:2,config:["p":.number(0.5)]))])
        _=try library.saveSource(.workflow(doc),id:doc.projectID)
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:doc.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        defer {owner.invalidate()}
        XCTAssertEqual(owner.document,doc)
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        defer {canvas.invalidate()}
        XCTAssertEqual(canvas.creationEntries.filter {$0.component.title=="Dropout"}.count,1)
        XCTAssertEqual(canvas.creationEntries.first {$0.component.title=="Dropout"}?.component.category,"Conditioning")
        _=owner.select(.init(nodeAddresses:[address]),lease:owner.lease())
        let context=try WorkflowModelNodeContext(owner:owner,address:address)
        let original=context.configuration
        XCTAssertFalse(context.setField("p",value:.number(1.2),expected:original))
        XCTAssertEqual(owner.document,doc)
        XCTAssertTrue(context.setField("p",value:.number(0.2),expected:original))
        XCTAssertFalse(context.setField("p",value:.number(0.8),expected:original))
        XCTAssertEqual(context.configuration["p"],.number(0.2))
        XCTAssertNotNil(owner.undo(lease:owner.lease()))
        XCTAssertEqual(owner.document.nodes,doc.nodes)
        XCTAssertEqual(owner.document.edges,doc.edges)
        XCTAssertGreaterThan(owner.document.revision,doc.revision)
    }
    func testEndpointTrainingGraphComparisonUsesTheSerializedSourceWithoutAcceptingChanges()throws {
        for p in [0.0,1.0,0.2] {
            let expected=GraphDocument(schemaVersion:2,nodes:[.init(id:"drop",type:"dropout",config:["p":.number(p)])],edges:[])
            let actual=try JSONDecoder().decode(GraphDocument.self,from:WorkflowPhaseArtifacts.encode(expected))
            XCTAssertTrue(try WorkflowPhaseArtifacts.matchesSerializedGraph(actual,expected:expected))
            var changed=actual;changed.nodes[0].config["p"] = .number(0.7)
            XCTAssertFalse(try WorkflowPhaseArtifacts.matchesSerializedGraph(changed,expected:expected))
            changed=actual;changed.nodes[0].config["p"] = .bool(true)
            XCTAssertFalse(try WorkflowPhaseArtifacts.matchesSerializedGraph(changed,expected:expected))
            changed=actual;changed.nodes[0].name="Different source"
            XCTAssertFalse(try WorkflowPhaseArtifacts.matchesSerializedGraph(changed,expected:expected))
        }
    }

}
