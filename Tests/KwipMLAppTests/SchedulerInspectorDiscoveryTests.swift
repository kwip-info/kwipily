import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class SchedulerInspectorDiscoveryTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func fixture(_ id:String,config:[String:JSONValue]=[:])throws->(WorkflowAuthoringStore,WorkflowNodeEditingAuthority,TrainInlineRecipeDraft) {
        try inlineRecipeFixture(id,role:"scheduler",config:config)
    }
    func testFiveSchedulersUseSharedDraftAuthorityAndPreserveApplyDiscardSemantics()throws {
        let keys=["scheduler_step_lr":"gamma","scheduler_cosine_annealing_lr":"eta_min","scheduler_reduce_on_plateau":"threshold","scheduler_one_cycle_lr":"max_lr","scheduler_linear_lr":"start_factor"]
        for id in SchedulerConfiguration.componentIDs.sorted() {
            let(owner,authority,context)=try fixture(id),key=try XCTUnwrap(keys[id])
            XCTAssertEqual(context.componentID,id)
            XCTAssertEqual(context.fields,BundledSchedulerComponents.definition(id).fields)
            XCTAssertEqual(context.draftConfiguration,try SchedulerConfiguration(componentID:id,values:[:]).values)
            let before=owner.document,origin=context.numericOrigin(key:key)
            XCTAssertTrue(context.setField(key,value:0.07,origin:context.numericOrigin(key:key)));XCTAssertTrue(context.hasChanges)
            XCTAssertTrue(owner.hasPendingInspectorEdits);XCTAssertEqual(owner.document,before)
            XCTAssertTrue(context.discard());XCTAssertEqual(owner.document,before);XCTAssertFalse(owner.hasPendingInspectorEdits)
            XCTAssertTrue(context.setField(key,value:0.08,origin:context.numericOrigin(key:key)));XCTAssertTrue(context.apply())
            XCTAssertEqual(context.committedConfiguration[key]?.doubleValue,0.08);XCTAssertTrue(authority.isCurrent)
            XCTAssertFalse(owner.hasPendingInspectorEdits);XCTAssertFalse(context.hasChanges)
            XCTAssertTrue(context.setField(key,value:0.09,origin:context.numericOrigin(key:key)));XCTAssertTrue(context.apply())
            XCTAssertEqual(context.committedConfiguration[key]?.doubleValue,0.09)
        }
    }
    func testOneCycleRetainsInactiveMomentumAndRejectsInvalidPhaseUntilRepaired()throws {
        let(owner,authority,context)=try fixture("scheduler_one_cycle_lr")
        XCTAssertTrue(context.setField("cycle_momentum",value:false,origin:context.numericOrigin(key:"cycle_momentum")))
        XCTAssertEqual(context.draftConfiguration["base_momentum"]?.doubleValue,0.85)
        XCTAssertTrue(context.apply())
        XCTAssertTrue(context.setField("pct_start",value:0.1,origin:context.numericOrigin(key:"pct_start")))
        let before=owner.document
        XCTAssertNotNil(context.validationMessage);XCTAssertFalse(context.apply());XCTAssertEqual(owner.document,before)
        XCTAssertTrue(owner.hasPendingInspectorEdits)
        XCTAssertTrue(context.setField("pct_start",value:0.4,origin:context.numericOrigin(key:"pct_start")))
        XCTAssertNil(context.validationMessage);XCTAssertTrue(context.apply())
        XCTAssertEqual(context.committedConfiguration["cycle_momentum"],false)
        XCTAssertFalse(context.setField("planned_total_batches",value:true,origin:context.numericOrigin(key:"planned_total_batches")))
        XCTAssertFalse(context.setField("last_epoch",value:2,origin:context.numericOrigin(key:"last_epoch")))
    }
    func testBusyForeignAndRetiredSchedulerDraftsCannotCommit()throws {
        let(owner,authority,context)=try fixture("scheduler_step_lr"),key="gamma"
        XCTAssertFalse(context.setField(key,value:0.2,origin:"foreign"))
        let origin=context.numericOrigin(key:key)
        context.setNumericPending(origin:origin,key:key,pending:true);XCTAssertTrue(owner.hasPendingInspectorEdits)
        owner.setExecutionBusy(true);XCTAssertFalse(context.apply());XCTAssertNil(context.currentNumericOrigin(key:key))
        owner.setExecutionBusy(false);context.setNumericPending(origin:origin,key:key,pending:false)
        XCTAssertTrue(context.setField(key,value:0.2,origin:context.numericOrigin(key:key)))
        let before=owner.document;authority.retire()
        XCTAssertFalse(owner.hasPendingInspectorEdits);XCTAssertFalse(context.apply());XCTAssertFalse(context.discard())
        XCTAssertEqual(owner.document,before)
    }
    func testInlineTrainSchedulersUseCanonicalDefaultsAndCompileEditsWithoutRecipeNodesOrWires()throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:directory)}
        let library=ProjectLibrary(directory:directory)
        let document=try WorkflowTrainRecipes.migrateLegacyRecipes(WorkflowExamples.preparedParallelClassifier(device:"cpu"))
        _=try library.saveSource(.workflow(document))
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repository.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertTrue(owner.select(.init(nodeAddresses:[WorkflowExamples.trainAddress]),lease:owner.lease()))
        let authority=try WorkflowNodeEditingAuthority(owner:owner,address:WorkflowExamples.trainAddress)
        let session=TrainInlineRecipesSession(authority:authority)
        let optimizer=TrainInlineRecipeDraft(session:session,role:"optimizer")
        XCTAssertTrue(optimizer.choose("optimizer_sgd"))
        XCTAssertTrue(optimizer.setField("lr",value:.number(0.2),origin:optimizer.numericOrigin("lr")))
        XCTAssertTrue(optimizer.apply(),optimizer.rejection ?? "")
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        XCTAssertFalse(canvas.creationEntries.contains {SchedulerConfiguration.componentIDs.contains($0.iconType)})
        XCTAssertEqual(Set(canvas.projection.ports.filter {$0.endpoint.address == WorkflowExamples.trainAddress && $0.direction == .input}.map {$0.endpoint.port}),["model","data","configuration"])
        let originalNodes=Set(owner.document.nodes.map(\.address))
        let keys=["scheduler_step_lr":"gamma","scheduler_cosine_annealing_lr":"eta_min","scheduler_reduce_on_plateau":"threshold","scheduler_one_cycle_lr":"max_lr","scheduler_linear_lr":"start_factor"]
        for id in SchedulerConfiguration.componentIDs.sorted() {
            let draft=TrainInlineRecipeDraft(session:session,role:"scheduler")
            XCTAssertTrue(draft.choose(id),id)
            let defaults=try SchedulerConfiguration(componentID:id,values:[:])
            XCTAssertEqual(draft.configuration,defaults.values,id)
            let envelope=try XCTUnwrap(draft.value())
            XCTAssertEqual(try SchedulerConfiguration.decodeRecipe(envelope),defaults)
            let key=try XCTUnwrap(keys[id]),before=owner.document
            XCTAssertTrue(draft.setField(key,value:.number(0.007),origin:draft.numericOrigin(key)))
            XCTAssertEqual(owner.document,before,"Inline numeric changes remain drafts until Apply")
            XCTAssertTrue(draft.apply(),draft.rejection ?? "")
            let expected=try SchedulerConfiguration(componentID:id,values:[key:.number(0.007)])
            let plan=try WorkflowCompiler.compile(owner.document,train:WorkflowExamples.trainAddress,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry,contractVersion:12)
            XCTAssertEqual(plan.schedulerRecipe?.configuration,expected,id)
            XCTAssertEqual(plan.schedulerRecipe?.source,WorkflowExamples.trainAddress)
            XCTAssertTrue(plan.inlineTrainingRecipeRoles.isSuperset(of:["optimizer","scheduler"]))
            XCTAssertEqual(Set(owner.document.nodes.map(\.address)),originalNodes)
            XCTAssertFalse(owner.document.nodes.contains {if case .schedulerRecipe=$0.payload {return true};return false})
            XCTAssertFalse(owner.document.edges.contains {$0.kind == .schedulerRecipe || $0.to.port == "scheduler"})
            draft.releasePendingEdits()
        }
    }
    func testOptionalExactDiscoveryAddsOnlyOneSchedulerAndResetRevokesIt()throws {
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let store=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
        let base=try JSONEncoder().encode(BuiltInCapabilityBridge.executionContract(registry:store.registry,runtimeVersion:"2.14.1",availableDevices:[.cpu]))
        let adam=try JSONEncoder().encode(BundledOptimizerComponents.entries["optimizer_adam"]!.runtimeContract)
        try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:base,optimizerDescriptors:["optimizer_adam":adam])
        XCTAssertTrue(store.schedulerRuntimeCapabilities.componentIDs.isEmpty);XCTAssertEqual(store.verifiedRuntimeComponentCount,6)
        for id in SchedulerConfiguration.componentIDs.sorted() {
            let exact=try JSONEncoder().encode(BundledSchedulerComponents.entries[id]!.runtimeContract)
            try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:base,optimizerDescriptors:["optimizer_adam":adam],schedulerDescriptors:[id:exact])
            XCTAssertEqual(store.schedulerRuntimeCapabilities.componentIDs,[id]);XCTAssertEqual(store.verifiedRuntimeComponentCount,7)
            var invalid=try XCTUnwrap(JSONSerialization.jsonObject(with:exact) as? [String:Any]);invalid["requiresExplicitOptimizer"]=false
            try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:base,optimizerDescriptors:["optimizer_adam":adam],schedulerDescriptors:[id:JSONSerialization.data(withJSONObject:invalid)])
            XCTAssertTrue(store.schedulerRuntimeCapabilities.componentIDs.isEmpty);XCTAssertEqual(store.verifiedRuntimeComponentCount,6)
        }
        try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:base)
        XCTAssertTrue(store.schedulerRuntimeCapabilities.componentIDs.isEmpty);XCTAssertEqual(store.verifiedRuntimeComponentCount,5)
    }
}
