import XCTest
import KwipMLCore
@testable import KwipMLApp

final class EmbeddingInspectorDraftTests:XCTestCase {
    func testOptionalDraftRetainsValuesWithoutPublishingAndRejectsInvalidNumbers()throws {
        let registry=try ComponentRegistry.builtIn()
        let defaults=Dictionary(uniqueKeysWithValues:try XCTUnwrap(registry.component(type:"embedding")).fields.map {($0.key,$0.defaultValue)})
        var draft=EmbeddingInspectorDraft(values:defaults)
        draft.setText("-1",key:"padding_idx");draft.paddingEnabled=true
        XCTAssertEqual(try draft.configuration(registry:registry)["padding_idx"],15)
        XCTAssertEqual(draft.values["padding_idx"],.null)
        draft.paddingEnabled=false
        XCTAssertEqual(try draft.configuration(registry:registry)["padding_idx"],.null)
        draft.normEnabled=true;draft.setText("1e-3",key:"max_norm")
        XCTAssertEqual(try draft.configuration(registry:registry)["max_norm"],.number(0.001))
        for invalid in ["0","nan","inf",""] {draft.setText(invalid,key:"max_norm");XCTAssertThrowsError(try draft.configuration(registry:registry))}
        draft.normEnabled=false;draft.setText("2.5",key:"num_embeddings")
        XCTAssertThrowsError(try draft.configuration(registry:registry))
    }
    func testConflictingGradientsAndTableBudgetAreRefused()throws {
        let registry=try ComponentRegistry.builtIn()
        var draft=EmbeddingInspectorDraft(values:Dictionary(uniqueKeysWithValues:try XCTUnwrap(registry.component(type:"embedding")).fields.map {($0.key,$0.defaultValue)}))
        draft.values["sparse"]=true;draft.values["scale_grad_by_freq"]=true
        XCTAssertThrowsError(try draft.configuration(registry:registry))
        draft.values["scale_grad_by_freq"]=false
        XCTAssertNoThrow(try draft.configuration(registry:registry))
        draft.setText("100000000",key:"num_embeddings");draft.setText("4096",key:"embedding_dim")
        XCTAssertThrowsError(try draft.configuration(registry:registry))
    }
    @MainActor func testOneToolbarEntryKeepsExistingSourceDefaultsAndOriginalLesson()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root),registry=try ComponentRegistry.builtIn()
        let defaults=Dictionary(uniqueKeysWithValues:try XCTUnwrap(registry.component(type:"embedding")).fields.map {($0.key,$0.defaultValue)})
        let address=WorkflowNodeAddress(domain:.model,localID:"interactions")
        let original=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,name:"Existing interactions",payload:.modelPrimitive(type:"embedding",configurationVersion:2,config:defaults))])
        _=try library.saveSource(.workflow(original))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:original.projectID,modelRegistry:registry,dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:registry,dataRegistry:owner.dataRegistry)
        defer {canvas.invalidate();owner.invalidate()}
        XCTAssertEqual(owner.document,original)
        let entry=try XCTUnwrap(canvas.creationEntries.first {$0.iconType=="embedding"})
        XCTAssertEqual(canvas.creationEntries.filter {$0.iconType=="embedding"}.count,1)
        guard case .modelPrimitive(let type,let version,let config)=entry.payload else {return XCTFail("Missing actual primitive")}
        XCTAssertEqual(type,"embedding");XCTAssertEqual(version,2)
        XCTAssertEqual(config,defaults)
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"embedding"),"embedding")
        XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID:"embedding"))
    }
    @MainActor func testWorkflowEditIsAtomicUndoableAndRejectsStaleSelection()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=try ComponentRegistry.builtIn(),library=ProjectLibrary(directory:root)
        let defaults=Dictionary(uniqueKeysWithValues:try XCTUnwrap(registry.component(type:"embedding")).fields.map {($0.key,$0.defaultValue)})
        let address=WorkflowNodeAddress(domain:.model,localID:"interactions")
        let config:[String:JSONValue]=defaults
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,name:"Interactions",payload:.modelPrimitive(type:"embedding",configurationVersion:2,config:config))])
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:registry,dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        defer {owner.invalidate()}
        XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
        let context=try WorkflowModelNodeContext(owner:owner,address:address)
        var next=config;next["num_embeddings"]=32;next["embedding_dim"]=6;next["padding_idx"]=0
        XCTAssertTrue(context.setEmbedding(configuration:next,expected:config),context.statusMessage)
        XCTAssertEqual(context.configuration,try EmbeddingConfiguration.validatedAuthoring(next,registry:registry))
        XCTAssertFalse(context.setEmbedding(configuration:config,expected:config))
        XCTAssertTrue(owner.undo(lease:owner.lease()));XCTAssertEqual(owner.document.nodes,document.nodes)
        XCTAssertTrue(owner.select(.init(),lease:owner.lease()))
        XCTAssertFalse(context.setEmbedding(configuration:next,expected:config))
        XCTAssertEqual(owner.document.nodes,document.nodes)
    }
}
