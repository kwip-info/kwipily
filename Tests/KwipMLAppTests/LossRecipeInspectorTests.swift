import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class LossRecipeInspectorTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func fixture(_ id:String,config:[String:JSONValue]=[:])throws->(WorkflowAuthoringStore,WorkflowNodeEditingAuthority,TrainInlineRecipeDraft) {
        try inlineRecipeFixture(id,role:"objective",config:config)
    }
    func testEightObjectivesUseActualCanonicalFieldsPublicInformationAndApplyDiscard()throws {
        for id in LossConfiguration.componentIDs.sorted() {
            let(owner,authority,context)=try fixture(id),definition=BundledLossComponents.definition(id)
            XCTAssertEqual(context.componentID,id);XCTAssertEqual(context.fields,definition.fields)
            XCTAssertEqual(context.draftConfiguration,try LossConfiguration(componentID:id,values:[:]).values)
            XCTAssertFalse(BundledLossComponents.entries[id]?.learning?.paragraphs.isEmpty ?? true)
            let before=owner.document,origin=context.numericOrigin(key:"reduction")
            XCTAssertTrue(context.setField("reduction",value:.string("sum"),origin:context.numericOrigin(key:"reduction")))
            XCTAssertTrue(context.hasChanges);XCTAssertTrue(owner.hasPendingInspectorEdits)
            XCTAssertEqual(owner.document,before,"Draft must not change source before Apply.")
            XCTAssertTrue(context.discard());XCTAssertEqual(owner.document,before);XCTAssertFalse(owner.hasPendingInspectorEdits)
            XCTAssertTrue(context.setField("reduction",value:.string("sum"),origin:context.numericOrigin(key:"reduction")));XCTAssertTrue(context.apply())
            XCTAssertEqual(context.committedConfiguration["reduction"],.string("sum"));XCTAssertEqual(context.expectedConfiguration,authority.configuration)
            XCTAssertFalse(context.hasChanges);XCTAssertFalse(owner.hasPendingInspectorEdits)
            XCTAssertTrue(context.setField("reduction",value:.string("none"),origin:context.numericOrigin(key:"reduction")));XCTAssertTrue(context.apply())
            XCTAssertEqual(context.committedConfiguration["reduction"],.string("none"))
        }
    }
    func testEightObjectivesAreTypedTrainChoicesAndLegacyRecipePaletteRemainsUnavailable()throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory)
        var legacy=WorkflowDocument(projectID:UUID().uuidString,nodes:[]);legacy.schemaVersion=2
        _=try library.saveSource(.workflow(legacy))
        let owner=try WorkflowAuthoringStore(library:library,projectID:legacy.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repository.appendingPathComponent("runner/dataflow-operators.json")))
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        defer { canvas.invalidate() }
        let entry=try XCTUnwrap(canvas.creationEntries.first {if case .train=$0.payload {return true};return false})
        canvas.addNode(type:entry.id,x:125,y:240)
        let train=try XCTUnwrap(owner.document.nodes.first {if case .train=$0.payload {return true};return false})
        XCTAssertEqual(owner.document.schemaVersion,18,"New Train placement owns current inline run settings and capture policy")
        XCTAssertEqual(train.x,125);XCTAssertEqual(train.y,240)
        XCTAssertTrue(owner.select(.init(nodeAddresses:[train.address]),lease:owner.lease()))
        let authority=try WorkflowNodeEditingAuthority(owner:owner,address:train.address)
        let session=TrainInlineRecipesSession(authority:authority)
        let draft=TrainInlineRecipeDraft(session:session,role:"objective")
        XCTAssertEqual(Set(TrainInlineRecipeDraft.IDs("objective")),LossConfiguration.componentIDs)
        for id in LossConfiguration.componentIDs.sorted() {
            XCTAssertFalse(canvas.creationEntries.contains {$0.iconType == id},"Objectives are configured on Train")
            XCTAssertTrue(draft.choose(id),id)
            XCTAssertEqual(draft.fields,BundledLossComponents.definition(id).fields)
            XCTAssertEqual(try draft.value(),try LossConfiguration(componentID:id,values:[:]).recipeValue)
            XCTAssertTrue(draft.apply(),draft.rejection ?? id)
            XCTAssertEqual(session.expected["objective"],try LossConfiguration(componentID:id,values:[:]).recipeValue)
            XCTAssertEqual(BundledLossComponents.entries[id]?.presentation.icon,id+"-v2")
            XCTAssertEqual(BundledLossComponents.entries[id]?.presentation.theme,"green")
        }
        XCTAssertGreaterThanOrEqual(owner.document.schemaVersion,13,"Applying inline objectives promotes the authoring schema")
        XCTAssertFalse(owner.document.nodes.contains {if case .lossRecipe=$0.payload {return true};return false})
        XCTAssertFalse(owner.document.edges.contains {$0.kind == .objectiveRecipe})
        XCTAssertNil(canvas.projection.portDescriptor(address:train.address,port:"objective",direction:.input))
    }
    func testBCEInterdependentWeightDraftRepairsAndRetainsInactiveConfiguredValues()throws {
        let(owner,authority,context)=try fixture("loss_bce_with_logits")
        func set(_ key:String,_ value:JSONValue) {XCTAssertTrue(context.setField(key,value:value,origin:context.numericOrigin(key:key)))}
        set("weight_mode",.string("configured"));set("weight_shape",.array([2]));set("weight_values",.array([0.25]))
        let before=owner.document
        XCTAssertNotNil(context.validationMessage);XCTAssertFalse(context.apply());XCTAssertEqual(owner.document,before)
        set("weight_values",.array([0.25,3]));XCTAssertNil(context.validationMessage);XCTAssertTrue(context.apply())
        for mode in ["none","mapped"] {
            set("weight_mode",.string(mode))
            XCTAssertEqual(context.draftConfiguration["weight_shape"],.array([2]))
            XCTAssertEqual(context.draftConfiguration["weight_values"],.array([.number(0.25),.number(3)]))
            XCTAssertTrue(context.apply());XCTAssertEqual(context.committedConfiguration["weight_values"],.array([.number(0.25),.number(3)]))
        }
        set("weight_mode",.string("configured"));XCTAssertTrue(context.apply())
        XCTAssertEqual(context.committedConfiguration["weight_values"],.array([.number(0.25),.number(3)]))
    }
    func testFull4096NumberListIsRetainedAndRecipeBudgetFailureRepairsWithoutTruncation()throws {
        let(owner,authority,context)=try fixture("loss_bce_with_logits")
        let field=try XCTUnwrap(context.fields.first {$0.key == "weight_values"}),origin=context.numericOrigin(key:"weight_values")
        let full=JSONValue.array(Array(repeating:.number(1),count:4096)),text=RegistryNumberListParsing.display(full)
        XCTAssertEqual(RegistryNumberListParsing.parse(text,field:field),full)
        var draft=RegistryNumberListDraft(value:context.draftConfiguration["weight_values"] ?? .array([]));draft.text="0.5, not-a-number"
        context.setNumericPending(origin:origin,key:"weight_values",pending:true)
        XCTAssertFalse(draft.commit(field:field,origin:origin,currentOrigin:context.currentNumericOrigin(key:"weight_values")) {context.setField("weight_values",value:$0,origin:origin)})
        XCTAssertTrue(owner.hasPendingInspectorEdits);XCTAssertFalse(context.canApply);XCTAssertEqual(draft.text,"0.5, not-a-number")
        draft.discard();context.setNumericPending(origin:origin,key:"weight_values",pending:false)
        XCTAssertTrue(context.setField("weight_mode",value:.string("configured"),origin:context.numericOrigin(key:"weight_mode")))
        XCTAssertTrue(context.setField("weight_shape",value:.array([4096]),origin:context.numericOrigin(key:"weight_shape")))
        XCTAssertTrue(context.setField("weight_values",value:full,origin:context.numericOrigin(key:"weight_values")));XCTAssertTrue(context.apply())
        XCTAssertEqual(context.committedConfiguration["weight_values"],full)
        let oversized=JSONValue.array(Array(repeating:.number(0.1234567890123456),count:4096)),before=owner.document
        XCTAssertTrue(context.setField("weight_values",value:oversized,origin:context.numericOrigin(key:"weight_values")))
        XCTAssertTrue(context.validationMessage?.contains("32768") ?? false)
        XCTAssertFalse(context.apply());XCTAssertEqual(owner.document,before);XCTAssertEqual(context.draftConfiguration["weight_values"],oversized)
        XCTAssertTrue(owner.hasPendingInspectorEdits)
        XCTAssertTrue(context.setField("weight_values",value:full,origin:context.numericOrigin(key:"weight_values")))
        XCTAssertNil(context.validationMessage);XCTAssertFalse(context.hasChanges)
        XCTAssertFalse(context.apply(),"Repairing exactly back to the committed baseline needs no new write")
        XCTAssertEqual(owner.document,before)
        XCTAssertFalse(owner.hasPendingInspectorEdits);XCTAssertEqual(context.committedConfiguration["weight_values"],full)
    }
    func testUndoBusyForeignAndRetiredDraftsCannotCommitIntoReplacement()throws {
        let(owner,authority,context)=try fixture("loss_l1"),origin=context.numericOrigin(key:"reduction")
        XCTAssertFalse(context.setField("reduction",value:.string("sum"),origin:"foreign"))
        context.setNumericPending(origin:origin,key:"reduction",pending:true)
        owner.setExecutionBusy(true);XCTAssertNil(context.currentNumericOrigin(key:"reduction"));XCTAssertFalse(context.apply())
        owner.setExecutionBusy(false);context.setNumericPending(origin:origin,key:"reduction",pending:false)
        XCTAssertTrue(context.setField("reduction",value:.string("sum"),origin:context.numericOrigin(key:"reduction")));XCTAssertTrue(context.apply())
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        XCTAssertFalse(authority.isCurrent);XCTAssertFalse(context.apply());XCTAssertFalse(context.discard())
        XCTAssertFalse(context.setField("reduction",value:.string("none"),origin:context.numericOrigin(key:"reduction")))
        guard case .train(let config)=owner.document.nodes[0].payload else {return XCTFail("Objective must remain Train-owned after Undo.")}
        XCTAssertEqual(config["objective"],try LossConfiguration(componentID:"loss_l1",values:[:]).recipeValue);XCTAssertFalse(owner.hasPendingInspectorEdits)
    }
    func testCrossEntropyProbabilityConstraintCanBeRepairedWithoutDisablingIgnoreControl()throws {
        let(owner,authority,context)=try fixture("loss_cross_entropy",config:["ignore_index":-7])
        XCTAssertTrue(context.setField("target_mode",value:.string("probabilities"),origin:context.numericOrigin(key:"target_mode")))
        XCTAssertNotNil(context.validationMessage);XCTAssertFalse(context.apply())
        XCTAssertTrue(context.setField("ignore_index",value:-100,origin:context.numericOrigin(key:"ignore_index")))
        XCTAssertNil(context.validationMessage);XCTAssertTrue(context.apply())
        XCTAssertEqual(context.committedConfiguration["ignore_index"],-100);XCTAssertFalse(owner.hasPendingInspectorEdits)
    }
}
