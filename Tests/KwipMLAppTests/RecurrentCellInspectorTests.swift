import XCTest
import KwipMLCore
@testable import KwipMLApp

final class RecurrentCellInspectorTests:XCTestCase {
    func testCellDraftRemembersRNNChoiceAndRejectsSequenceSettings()throws {
        let registry=try ComponentRegistry.builtIn()
        var draft=RecurrentInspectorDraft(method:.rnn,values:try RecurrentCellConfiguration.defaults(for:.rnn,registry:registry),isCell:true)
        draft.setText("8",key:"input_size");draft.setText("16",key:"hidden_size")
        draft.values["nonlinearity"] = .string("relu");draft.values["state_mode"] = .string("supplied")
        try draft.select(.lstm,registry:registry)
        XCTAssertTrue(draft.isCell);XCTAssertEqual(draft.values["hidden_size"],16);XCTAssertNil(draft.values["nonlinearity"])
        try draft.select(.gru,registry:registry);try draft.select(.rnn,registry:registry)
        XCTAssertEqual(draft.values["nonlinearity"],.string("relu"));XCTAssertEqual(draft.values["state_mode"],.string("supplied"))
        for key in ["num_layers","dropout","proj_size"] {
            var invalid=draft;invalid.setText("1",key:key);XCTAssertThrowsError(try invalid.configuration(registry:registry))
        }
        draft.setText("0",key:"input_size");XCTAssertThrowsError(try draft.configuration(registry:registry))
        draft.setText("1.5",key:"input_size");XCTAssertThrowsError(try draft.configuration(registry:registry))
    }
    @MainActor func testPaletteAndInsertionConsolidateOnlyEligibleCellMethods()throws {
        let registry=try ComponentRegistry.builtIn()
        func entries(_ ids:Set<String>)->[ComponentRibbonEntry] {registry.components.filter {ids.contains($0.id)}.map {.init(id:$0.id,title:$0.title,shortTitle:$0.title,purpose:"",category:$0.category,typeIcon:$0.id)}}
        let offered=ComponentRibbonCatalog.visible(entries(RecurrentConfiguration.componentIDs),scope:.all,query:"")
        XCTAssertEqual(Set(offered.map(\.typeIcon)),[RecurrentPalette.type,RecurrentCellPalette.type])
        let limited=ComponentRibbonCatalog.visible(entries(["lstm_cell"]),scope:.all,query:"LSTM")
        XCTAssertEqual(limited.map(\.id),["lstm_cell"]);XCTAssertEqual(limited.map(\.typeIcon),[RecurrentCellPalette.type])
        XCTAssertTrue(ComponentRibbonCatalog.visible(entries(["lstm_cell"]),scope:.all,query:"GRU").isEmpty)
        XCTAssertEqual(RecurrentCellPalette.defaultMethod(allowedTypes:["lstm_cell"]),.lstm)
        let insertion=RecurrentCellPalette.consolidated(components:registry.components.filter {["lstm_cell","gru_cell"].contains($0.id)})
        XCTAssertEqual(insertion.map(\.id),["lstm_cell"])
        XCTAssertEqual(ComponentLessonDestination.entryID(type:RecurrentCellPalette.type),"recurrent-cell-overview")
        for id in RecurrentCellConfiguration.componentIDs {XCTAssertNotNil(ComponentLessonDestination.entryID(type:id))}
        XCTAssertEqual(TrainLayerPresentation.parameterTitle("weight_ih"),"Input weights")
        XCTAssertEqual(TrainLayerPresentation.parameterTitle("bias_hh"),"Recurrent biases")
    }
    @MainActor func testWorkflowCellEditPreservesSourceAndRejectsStaleOwner()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=try ComponentRegistry.builtIn(),library=ProjectLibrary(directory:root)
        let address=WorkflowNodeAddress(domain:.model,localID:"memory")
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,name:"Step",payload:.modelPrimitive(type:"rnn_cell",configurationVersion:2,config:[:]))])
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:registry,dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:registry,dataRegistry:owner.dataRegistry)
        defer {canvas.invalidate();owner.invalidate()}
        XCTAssertEqual(owner.document,document)
        XCTAssertEqual(canvas.creationEntries.filter {$0.iconType==RecurrentCellPalette.type}.count,1)
        XCTAssertFalse(canvas.creationEntries.contains {RecurrentCellConfiguration.componentIDs.contains($0.iconType)})
        XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
        let context=try WorkflowModelNodeContext(owner:owner,address:address)
        let config=try RecurrentCellConfiguration.defaults(for:.gru,registry:registry)
        XCTAssertTrue(context.setRecurrent(.gru,configuration:config,expectedType:"rnn_cell",expected:[:]),context.statusMessage)
        XCTAssertEqual(try RecurrentCellConfiguration.configuration(of:owner.document.nodes[0],registry:registry).method,.gru)
        XCTAssertFalse(context.setRecurrent(.lstm,configuration:config,expectedType:"rnn_cell",expected:[:]))
        XCTAssertTrue(owner.undo(lease:owner.lease()));XCTAssertEqual(owner.document.nodes,document.nodes)
        XCTAssertTrue(owner.select(.init(),lease:owner.lease()))
        XCTAssertFalse(context.setRecurrent(.gru,configuration:config,expectedType:"rnn_cell",expected:[:]))
    }
    @MainActor func testExposedCellPortsRefuseRemovalWithoutMutatingOwner()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=try ComponentRegistry.builtIn()
        var config=try RecurrentCellConfiguration.defaults(for:.lstm,registry:registry);config["state_mode"] = .string("supplied")
        let definition=CompositeDefinition(reference:.init("cell-owner"),title:"Step",body:.init(nodes:[.init(id:"memory",type:"lstm_cell",config:config)],configurationVersion:2),inputs:[.init("in",targets:[.init("memory","in")]),.init("hidden",targets:[.init("memory","hidden")]),.init("cell",targets:[.init("memory","cell")])],outputs:[.init("out",source:.init("memory","out")),.init("cell",source:.init("memory","cell"))])
        let source=CompositeProjectDocument(name:"Step",root:.init(instances:[.init(id:"group",definition:definition.reference)]),definitions:[definition])
        let library=ProjectLibrary(directory:root),id=UUID().uuidString
        _=try library.saveDocument(.blockModel(.init(projectID:id,source:source)))
        let owner=try ModelAuthoringStore(library:library,projectID:id)
        XCTAssertTrue(owner.beginDraft(path:["group"],lease:owner.lease()))
        let before=owner.document
        XCTAssertNil(owner.applyDraftCommand(.setRecurrent(nodeID:"memory",method:.gru),lease:owner.lease(path:["group"])))
        XCTAssertEqual(owner.document,before)
        config["state_mode"] = .string("zero")
        XCTAssertNil(owner.applyDraftCommand(.setRecurrent(nodeID:"memory",method:.lstm,configuration:config),lease:owner.lease(path:["group"])))
        XCTAssertEqual(owner.document,before)
    }
    @MainActor func testScopedBoundCellEditDoesNotMistakeMethodForSequenceType()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let definition=CompositeDefinition(reference:.init("cell-bound-owner"),title:"Step",body:.init(nodes:[.init(id:"memory",type:"rnn_cell",config:[:])],configurationVersion:2),inputs:[.init("in",targets:[.init("memory","in")])],outputs:[.init("out",source:.init("memory","out"))],parameters:[.init(.init(key:"width",title:"Width",defaultValue:8,min:1,max:512),targets:[.init("memory","hidden_size")])])
        let source=CompositeProjectDocument(name:"Step",root:.init(instances:[.init(id:"group",definition:definition.reference)]),definitions:[definition])
        let library=ProjectLibrary(directory:root),id=UUID().uuidString
        _=try library.saveDocument(.blockModel(.init(projectID:id,source:source)))
        let owner=try ModelAuthoringStore(library:library,projectID:id)
        XCTAssertTrue(owner.beginDraft(path:["group"],lease:owner.lease()))
        var values=try RecurrentCellConfiguration.defaults(for:.rnn,registry:.builtIn());values["hidden_size"]=8;values["bias"]=false
        XCTAssertNotNil(owner.applyDraftCommand(.setRecurrent(nodeID:"memory",method:.rnn,configuration:values),lease:owner.lease(path:["group"])),owner.error ?? "")
        let current=owner.document
        XCTAssertNil(owner.applyDraftCommand(.setRecurrent(nodeID:"memory",method:.gru),lease:owner.lease(path:["group"])))
        XCTAssertEqual(owner.document,current)
    }

}
