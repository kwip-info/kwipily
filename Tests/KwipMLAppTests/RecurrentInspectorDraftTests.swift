import XCTest
import KwipMLCore
@testable import KwipMLApp

final class RecurrentInspectorDraftTests:XCTestCase {
    func testSharedControlsAndMethodSpecificDraftsArePreservedWithoutPublishing()throws {
        let registry=try ComponentRegistry.builtIn()
        var draft=RecurrentInspectorDraft(method:.rnn,values:try RecurrentSequenceConfiguration.defaults(for:.rnn,registry:registry))
        draft.setText("8",key:"input_size");draft.setText("16",key:"hidden_size");draft.setText("2",key:"num_layers");draft.setText("1.5e-1",key:"dropout")
        draft.values["nonlinearity"] = .string("relu");draft.values["state_mode"] = .string("supplied");draft.values["bidirectional"]=true
        XCTAssertEqual(draft.values["input_size"],2)
        try draft.select(.lstm,registry:registry)
        XCTAssertEqual(draft.values["input_size"],8);XCTAssertEqual(draft.values["hidden_size"],16)
        XCTAssertEqual(draft.values["dropout"],.number(0.15));XCTAssertNil(draft.values["nonlinearity"])
        draft.setText("8",key:"proj_size");try draft.select(.gru,registry:registry)
        XCTAssertNil(draft.values["proj_size"]);XCTAssertEqual(draft.values["state_mode"],.string("supplied"))
        try draft.select(.lstm,registry:registry);XCTAssertEqual(draft.values["proj_size"],8)
        try draft.select(.rnn,registry:registry);XCTAssertEqual(draft.values["nonlinearity"],.string("relu"))
    }
    func testDropoutLayerAndProjectionInvalidDraftsRequireExplicitRepair()throws {
        let registry=try ComponentRegistry.builtIn()
        var draft=RecurrentInspectorDraft(method:.lstm,values:try RecurrentSequenceConfiguration.defaults(for:.lstm,registry:registry))
        draft.setText("nan",key:"dropout");XCTAssertThrowsError(try draft.configuration(registry:registry))
        draft.setText("0.2",key:"dropout");XCTAssertThrowsError(try draft.configuration(registry:registry))
        draft.setText("2",key:"num_layers");XCTAssertNoThrow(try draft.configuration(registry:registry))
        draft.setText("1",key:"num_layers");XCTAssertThrowsError(try draft.configuration(registry:registry))
        draft.setText("0",key:"dropout");XCTAssertNoThrow(try draft.configuration(registry:registry))
        draft.setText("2",key:"proj_size");XCTAssertThrowsError(try draft.configuration(registry:registry))
        draft.setText("1",key:"proj_size");XCTAssertNoThrow(try draft.configuration(registry:registry))
        draft.setText("1.5",key:"hidden_size");XCTAssertThrowsError(try draft.configuration(registry:registry))
    }
    @MainActor func testPaletteOffersSequenceAndCellFamiliesAndRestrictedMethods()throws {
        let registry=try ComponentRegistry.builtIn()
        func entries(_ ids:Set<String>)->[ComponentRibbonEntry] {registry.components.filter {ids.contains($0.id)}.map {.init(id:$0.id,title:$0.title,shortTitle:$0.title,purpose:"",category:$0.category,typeIcon:$0.id)}}
        let values=ComponentRibbonCatalog.visible(entries(RecurrentConfiguration.componentIDs),scope:.all,query:"")
        XCTAssertEqual(values.count,2)
        XCTAssertEqual(values.filter {$0.typeIcon=="kwip.recurrent"}.count,1)
        XCTAssertFalse(values.contains {RecurrentSequenceConfiguration.componentIDs.contains($0.typeIcon)})
        XCTAssertEqual(values.filter {$0.typeIcon==RecurrentCellPalette.type}.count,1)
        let limited=ComponentRibbonCatalog.visible(entries(["lstm"]),scope:.all,query:"LSTM")
        XCTAssertEqual(limited.map(\.id),["lstm"]);XCTAssertEqual(limited.map(\.typeIcon),["kwip.recurrent"])
        XCTAssertTrue(ComponentRibbonCatalog.visible(entries(["lstm"]),scope:.all,query:"GRU").isEmpty)
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"kwip.recurrent"),"recurrent-overview")
        for id in ["rnn","lstm","gru"] {XCTAssertNotNil(ComponentLessonDestination.entryID(type:id))}
    }
    func testFriendlyParameterNamesKeepLayerDirectionAndRoleDistinct() {
        XCTAssertEqual(TrainLayerPresentation.parameterTitle("weight_ih_l0"),"Input weights · Layer 1 · Forward")
        XCTAssertEqual(TrainLayerPresentation.parameterTitle("weight_hh_l1_reverse"),"Recurrent weights · Layer 2 · Reverse")
        XCTAssertEqual(TrainLayerPresentation.parameterTitle("weight_hr_l0"),"Projection weights · Layer 1 · Forward")
        XCTAssertEqual(TrainLayerPresentation.parameterTitle("bias_hh_l0"),"Recurrent biases · Layer 1 · Forward")
        XCTAssertEqual(TrainLayerPresentation.parameterTitle("weight"),"Weights")
    }
    @MainActor func testOldSourcesStayExactAndOnlyOneActualDefaultCreationAppears()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let library=ProjectLibrary(directory:root),registry=try ComponentRegistry.builtIn()
        let nodes=RecurrentSequenceConfiguration.Method.allCases.map {WorkflowNode(address:.init(domain:.model,localID:$0.rawValue),name:$0.title,payload:.modelPrimitive(type:$0.rawValue,configurationVersion:2,config:[:]))}
        let original=WorkflowDocument(projectID:UUID().uuidString,nodes:nodes)
        _=try library.saveSource(.workflow(original))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:original.projectID,modelRegistry:registry,dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:registry,dataRegistry:owner.dataRegistry)
        defer {canvas.invalidate();owner.invalidate()}
        XCTAssertEqual(owner.document,original)
        XCTAssertEqual(canvas.creationEntries.filter {$0.iconType=="kwip.recurrent"}.count,1)
        XCTAssertFalse(canvas.creationEntries.contains {RecurrentSequenceConfiguration.componentIDs.contains($0.iconType)})
        XCTAssertEqual(canvas.creationEntries.filter {$0.iconType==RecurrentCellPalette.type}.count,1)
    }
}
