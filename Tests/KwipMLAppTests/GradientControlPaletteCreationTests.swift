import XCTest
import KwipMLCore
@testable import KwipMLApp

final class GradientControlPaletteCreationTests:XCTestCase {
    @MainActor private func checkInlineControl(_ componentID:String)throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let models=try ComponentRegistry.builtIn()
        let data=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        var legacy=try WorkflowExamples.preparedParallelClassifier(device:"cpu");legacy.schemaVersion=9
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(legacy)),legacy)
        let library=ProjectLibrary(directory:directory)
        _=try library.saveSource(.workflow(legacy))
        let owner=try WorkflowAuthoringStore(library:library,projectID:legacy.projectID,modelRegistry:models,dataRegistry:data)
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:models,dataRegistry:data)
        XCTAssertFalse(canvas.creationEntries.contains {$0.iconType==componentID},"Configuration-only controls belong to Train, not current palette placement.")
        XCTAssertTrue(owner.select(.init(nodeAddresses:[WorkflowExamples.trainAddress]),lease:owner.lease()))
        let session=TrainInlineRecipesSession(authority:try WorkflowNodeEditingAuthority(owner:owner,address:WorkflowExamples.trainAddress))
        let role=componentID=="gradient_policy" ? "gradientPolicy":"trainability"
        let draft=TrainInlineRecipeDraft(session:session,role:role)
        let before=owner.document
        let legacyTrain=try XCTUnwrap(before.nodes.first {$0.address==WorkflowExamples.trainAddress})
        let legacyPorts=try WorkflowPortContracts.resolve(legacyTrain,document:before,modelRegistry:models,dataRegistry:data)
        XCTAssertEqual(legacyPorts.inputs.map(\.id),["model","data","configuration"],"Current authoring exposes three Train inputs even for an older-schema document.")
        let currentTrainEntry=try XCTUnwrap(canvas.creationEntries.first {if case .train=$0.payload {return true};return false})
        XCTAssertEqual(currentTrainEntry.component.inputs,["model","training","validation"],"New placement exposes only current Train inputs.")
        let baseline=try WorkflowCompiler.compile(before,train:WorkflowExamples.trainAddress,modelRegistry:models,dataRegistry:data,contractVersion:13)
        XCTAssertTrue(draft.choose(componentID))
        var selectedOwner:InitializationOwnerChoice?
        if componentID=="gradient_policy" {
            XCTAssertTrue(try XCTUnwrap(draft.gradient).setAccumulationSteps(3))
        } else {
            selectedOwner=try XCTUnwrap(session.modelChoices().owners.first)
            let editor=try XCTUnwrap(draft.trainability)
            XCTAssertTrue(editor.addOwner(try XCTUnwrap(selectedOwner).id))
            XCTAssertTrue(editor.setMode("frozen",at:0))
        }
        XCTAssertTrue(owner.hasPendingInspectorEdits)
        XCTAssertTrue(draft.apply(),draft.validationMessage ?? draft.rejection ?? "")
        XCTAssertFalse(owner.hasPendingInspectorEdits)
        XCTAssertEqual(owner.document.nodes.map(\.address),before.nodes.map(\.address))
        XCTAssertEqual(owner.document.edges,before.edges)
        XCTAssertEqual(owner.document.revision,before.revision+1)
        let train=try XCTUnwrap(owner.document.nodes.first {$0.address==WorkflowExamples.trainAddress})
        let ports=try WorkflowPortContracts.resolve(train,document:owner.document,modelRegistry:models,dataRegistry:data)
        XCTAssertGreaterThanOrEqual(owner.document.schemaVersion,13)
        XCTAssertEqual(ports.inputs.map(\.id),["model","data","configuration"],"Inline policy settings do not introduce recipe execution ports.")
        let displayedID=try XCTUnwrap(canvas.projection.displayIDByAddress[WorkflowExamples.trainAddress])
        let displayedTrain=try XCTUnwrap(canvas.graph.nodes.first {$0.id==displayedID})
        XCTAssertEqual(try XCTUnwrap(canvas.registry.component(type:displayedTrain.type)).inputs,["model","data","configuration"],"Current canvas presentation hides compatibility recipe ports.")
        guard case .train(let values)=train.payload else {return XCTFail("Missing Train configuration")}
        let recipes=try WorkflowTrainRecipes(configuration:values,schemaVersion:owner.document.schemaVersion)
        let plan=try WorkflowCompiler.compile(owner.document,train:WorkflowExamples.trainAddress,modelRegistry:models,dataRegistry:data,contractVersion:13)
        XCTAssertEqual(plan.model.graph,baseline.model.graph,"Editing a training policy must not change the executable model.")
        XCTAssertEqual(plan.model.semanticSHA256,baseline.model.semanticSHA256)
        XCTAssertEqual(plan.model.origins,baseline.model.origins)
        if componentID=="gradient_policy" {
            XCTAssertEqual(recipes.gradientPolicy?.accumulationSteps,3)
            XCTAssertEqual(plan.gradientPolicyRecipe?.configuration,recipes.gradientPolicy)
            XCTAssertEqual(plan.gradientPolicyRecipe?.source,WorkflowExamples.trainAddress)
        } else {
            let selected=try XCTUnwrap(selectedOwner)
            XCTAssertEqual(recipes.trainability?.owners,[.object(["node":.string(selected.node),"parameter":.string(selected.parameter),"mode":.string("frozen")])])
            XCTAssertEqual(plan.trainabilityRecipe?.configuration,recipes.trainability)
            XCTAssertEqual(plan.trainabilityRecipe?.source,WorkflowExamples.trainAddress)
        }
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(owner.document)),owner.document)
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        var restored=before;restored.revision=before.revision+2
        XCTAssertEqual(owner.document,restored)
        XCTAssertFalse(draft.apply(),"Undo retires the editor's previous authority.")
    }
    @MainActor func testGradientPolicyIsEditedInlineAndLegacyModelSurvivesUndo()throws {try checkInlineControl("gradient_policy")}
    @MainActor func testTrainabilityIsEditedInlineWithExactOwnerAndUndo()throws {try checkInlineControl("trainability")}
}
