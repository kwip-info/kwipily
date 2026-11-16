import XCTest
import KwipMLCore
@testable import KwipMLApp

final class BilinearInspectorDraftTests:XCTestCase {
    func testDraftKeepsIndependentWidthsUnpublishedAndCountsRealParameters()throws {
        var draft=BilinearInspectorDraft(values:["in1_features":2,"in2_features":2,"out_features":2,"bias":true])
        draft.setText("3",key:"in1_features");draft.setText("5",key:"in2_features")
        XCTAssertEqual(draft.values["in1_features"],2)
        XCTAssertEqual(try BilinearConfiguration(values:draft.configuration()).parameterCount,32)
        draft.values["bias"]=false
        XCTAssertEqual(try BilinearConfiguration(values:draft.configuration()).parameterCount,30)
        for key in ["in1_features","in2_features","out_features"] {
            let before=draft
            for bad in ["0","513","2.5","nan",""] {
                draft.setText(bad,key:key);XCTAssertThrowsError(try draft.configuration())
            }
            draft=before
        }
        for key in ["in1_features","in2_features","out_features"] {draft.setText("512",key:key)}
        XCTAssertThrowsError(try draft.configuration(),"All individually valid widths can exceed the allocation ceiling together")
    }
    @MainActor func testOneToolbarEntryKeepsExistingSourceDefaultsAndOriginalLesson()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root),registry=try ComponentRegistry.builtIn()
        let address=WorkflowNodeAddress(domain:.model,localID:"interactions")
        let original=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,name:"Existing interactions",payload:.modelPrimitive(type:"bilinear",configurationVersion:2,config:["in1_features":3,"in2_features":5,"out_features":2,"bias":false]))])
        _=try library.saveSource(.workflow(original))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:original.projectID,modelRegistry:registry,dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:registry,dataRegistry:owner.dataRegistry)
        defer {canvas.invalidate();owner.invalidate()}
        XCTAssertEqual(owner.document,original)
        let entry=try XCTUnwrap(canvas.creationEntries.first {$0.iconType=="bilinear"})
        XCTAssertEqual(canvas.creationEntries.filter {$0.iconType=="bilinear"}.count,1)
        guard case .modelPrimitive(let type,let version,let config)=entry.payload else {return XCTFail("Missing actual primitive")}
        XCTAssertEqual(type,"bilinear");XCTAssertEqual(version,2)
        XCTAssertEqual(config,["in1_features":2,"in2_features":2,"out_features":2,"bias":true])
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"bilinear"),"bilinear")
        XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID:"bilinear"))
    }
    @MainActor func testWorkflowEditIsAtomicUndoableAndRejectsStaleSelection()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=try ComponentRegistry.builtIn(),library=ProjectLibrary(directory:root)
        let address=WorkflowNodeAddress(domain:.model,localID:"interactions")
        let config:[String:JSONValue]=["in1_features":3,"in2_features":5,"out_features":2,"bias":true]
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,name:"Interactions",payload:.modelPrimitive(type:"bilinear",configurationVersion:2,config:config))])
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:registry,dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        defer {owner.invalidate()}
        XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
        let context=try WorkflowModelNodeContext(owner:owner,address:address)
        var next=config;next["in1_features"]=4;next["in2_features"]=6;next["bias"]=false
        XCTAssertTrue(context.setBilinear(configuration:next,expected:config),context.statusMessage)
        XCTAssertEqual(context.configuration,next)
        XCTAssertFalse(context.setBilinear(configuration:config,expected:config))
        XCTAssertTrue(owner.undo(lease:owner.lease()));XCTAssertEqual(owner.document.nodes,document.nodes)
        XCTAssertTrue(owner.select(.init(),lease:owner.lease()))
        XCTAssertFalse(context.setBilinear(configuration:next,expected:config))
        XCTAssertEqual(owner.document.nodes,document.nodes)
    }
}
