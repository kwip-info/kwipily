import XCTest
import KwipMLCore
@testable import KwipMLApp
final class EmbeddingBagInspectorDraftTests:XCTestCase {
    func testLayoutsAndWeightsRemainExplicitAndInvalidDraftsDoNotPublish()throws {
        let registry=try ComponentRegistry.builtIn()
        var draft=EmbeddingBagInspectorDraft(values:EmbeddingBagConfiguration.defaults)
        draft.values["layout"] = .string("packed");draft.values["include_last_offset"]=true
        draft.values["mode"] = .string("sum");draft.values["per_sample_weights"]=true
        XCTAssertEqual(try EmbeddingBagConfiguration(values:draft.configuration(registry:registry)).inputPorts,["indices","offsets","per_sample_weights"])
        draft.values["layout"] = .string("fixed")
        XCTAssertEqual(try draft.configuration(registry:registry)["include_last_offset"],true,"Hidden terminal policy is preserved")
        draft.values["mode"] = .string("max")
        XCTAssertThrowsError(try draft.configuration(registry:registry))
        XCTAssertEqual(draft.values["per_sample_weights"],true,"Do not silently discard a weighted input")
        draft.values["per_sample_weights"]=false;draft.values["scale_grad_by_freq"]=true
        XCTAssertThrowsError(try draft.configuration(registry:registry))
        draft.values["mode"] = .string("mean");draft.values["sparse"]=true
        XCTAssertThrowsError(try draft.configuration(registry:registry))
    }
    func testOptionalNumbersAndFloat32Limits()throws {
        let registry=try ComponentRegistry.builtIn()
        var draft=EmbeddingBagInspectorDraft(values:EmbeddingBagConfiguration.defaults)
        draft.paddingEnabled=true;draft.setText("-1",key:"padding_idx")
        draft.normEnabled=true;draft.setText("1e-3",key:"max_norm")
        let values=try draft.configuration(registry:registry)
        XCTAssertEqual(values["padding_idx"],15);XCTAssertEqual(values["max_norm"],.number(0.001))
        for text in ["nan","0","-1","1e-100", "1e100",""] {draft.setText(text,key:"max_norm");XCTAssertThrowsError(try draft.configuration(registry:registry))}
        draft.normEnabled=false;XCTAssertEqual(try draft.configuration(registry:registry)["max_norm"],.null)
        draft.setText("1.5",key:"num_embeddings");XCTAssertThrowsError(try draft.configuration(registry:registry))
    }
    @MainActor func testOneToolbarEntryKeepsExistingSourceDefaultsAndOriginalLesson()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root),registry=try ComponentRegistry.builtIn()
        let defaults=Dictionary(uniqueKeysWithValues:try XCTUnwrap(registry.component(type:"embedding_bag")).fields.map {($0.key,$0.defaultValue)})
        let address=WorkflowNodeAddress(domain:.model,localID:"interactions")
        let original=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,name:"Existing interactions",payload:.modelPrimitive(type:"embedding_bag",configurationVersion:1,config:defaults))])
        _=try library.saveSource(.workflow(original))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:original.projectID,modelRegistry:registry,dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:registry,dataRegistry:owner.dataRegistry)
        defer {canvas.invalidate();owner.invalidate()}
        XCTAssertEqual(owner.document,original)
        let entry=try XCTUnwrap(canvas.creationEntries.first {$0.iconType=="embedding_bag"})
        XCTAssertEqual(canvas.creationEntries.filter {$0.iconType=="embedding_bag"}.count,1)
        guard case .modelPrimitive(let type,let version,let config)=entry.payload else {return XCTFail("Missing actual primitive")}
        XCTAssertEqual(type,"embedding_bag");XCTAssertEqual(version,2)
        XCTAssertEqual(config,defaults)
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"embedding_bag"),"embedding-bag")
        XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID:"embedding-bag"))
    }
    @MainActor func testWorkflowEditIsAtomicUndoableAndRejectsStaleSelection()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=try ComponentRegistry.builtIn(),library=ProjectLibrary(directory:root)
        let defaults=Dictionary(uniqueKeysWithValues:try XCTUnwrap(registry.component(type:"embedding_bag")).fields.map {($0.key,$0.defaultValue)})
        let address=WorkflowNodeAddress(domain:.model,localID:"interactions")
        let config:[String:JSONValue]=defaults
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,name:"Interactions",payload:.modelPrimitive(type:"embedding_bag",configurationVersion:1,config:config))])
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:registry,dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        defer {owner.invalidate()}
        XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
        let context=try WorkflowModelNodeContext(owner:owner,address:address)
        var next=config;next["num_embeddings"]=32;next["embedding_dim"]=6;next["padding_idx"]=0
        XCTAssertTrue(context.setEmbeddingBag(configuration:next,expected:config),context.statusMessage)
        XCTAssertEqual(context.configuration,try EmbeddingBagConfiguration.validatedAuthoring(next,registry:registry))
        XCTAssertFalse(context.setEmbeddingBag(configuration:config,expected:config))
        XCTAssertTrue(owner.undo(lease:owner.lease()));XCTAssertEqual(owner.document.nodes,document.nodes)
        XCTAssertTrue(owner.select(.init(),lease:owner.lease()))
        XCTAssertFalse(context.setEmbeddingBag(configuration:next,expected:config))
        XCTAssertEqual(owner.document.nodes,document.nodes)
    }
}
