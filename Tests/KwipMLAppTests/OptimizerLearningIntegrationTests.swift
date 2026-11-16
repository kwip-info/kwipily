import XCTest
import KwipMLCore
@testable import KwipMLApp

final class OptimizerLearningIntegrationTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    @MainActor func testRecipeCanvasHasTypedControlsNoTensorBoundariesAndPreservesDraftEdits()throws {
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),lesson=TensorLearningStore(workspace:workspace)
        for task in OptimizerLearningTask.allCases {
            XCTAssertTrue(lesson.start(operation:task.rawValue));let canvas=try XCTUnwrap(lesson.canvas)
            XCTAssertEqual(Set(canvas.registry.components.map(\.id)),[task.componentID,"train"])
            XCTAssertFalse(canvas.canAddType("input"));canvas.addNode(type:task.componentID,x:0,y:0);canvas.addNode(type:"train",x:200,y:0)
            XCTAssertEqual(canvas.graph.schemaVersion,6)
            let recipe=try XCTUnwrap(canvas.graph.nodes.first {$0.type == task.componentID}),train=try XCTUnwrap(canvas.graph.nodes.first {$0.type == "train"})
            XCTAssertTrue(canvas.canConnect(source:recipe.id,sourcePort:"out",target:train,port:"optimizer"));canvas.connect(source:recipe.id,sourcePort:"out",target:train.id,targetPort:"optimizer")
            XCTAssertTrue(canvas.validation.isValid);XCTAssertTrue(canvas.validation.tensorLayouts.isEmpty)
            canvas.undo();XCTAssertTrue(canvas.graph.edges.isEmpty);XCTAssertEqual(canvas.graph.schemaVersion,6);canvas.redo();XCTAssertEqual(canvas.graph.edges.count,1);XCTAssertEqual(canvas.graph.schemaVersion,6)
            canvas.moveNodes(ids:[recipe.id],dx:40,dy:20);XCTAssertEqual(canvas.graph.nodes.first {$0.id == recipe.id}?.x,40);canvas.undo();XCTAssertEqual(canvas.graph.nodes.first {$0.id == recipe.id}?.x,0);XCTAssertEqual(canvas.graph.schemaVersion,6)
            XCTAssertFalse(canvas.canvasPortDescription(nodeID:recipe.id,port:"out").contains("Shape unresolved"))
            XCTAssertTrue(canvas.canvasPortDescription(nodeID:train.id,port:"optimizer").lowercased().contains("recipe"))
            XCTAssertFalse(canvas.canAddType(task.componentID));XCTAssertFalse(canvas.canAddType("train"))
            XCTAssertFalse(canvas.initializeNormalizationParameters(nodeID:train.id))
            XCTAssertTrue(canvas.setPending(nodeID:recipe.id,field:"lr",text:"0.2"));XCTAssertNotNil(canvas.pendingText(nodeID:recipe.id,field:"lr"));XCTAssertThrowsError(try lesson.draft?.submissionData())
            let saved=try XCTUnwrap(lesson.draft).encoded();let restored=try TensorLearningDraft.decode(saved);XCTAssertFalse(restored.pendingFields.isEmpty)
            canvas.discardPending(nodeID:recipe.id,field:"lr");XCTAssertNil(canvas.pendingText(nodeID:recipe.id,field:"lr"))
            var c=recipe.config;c["lr"] = .number(0.2);XCTAssertTrue(canvas.configure(nodeID:recipe.id,values:c));canvas.undo();XCTAssertEqual(canvas.graph.nodes.first {$0.id == recipe.id}?.config["lr"]?.doubleValue,recipe.config["lr"]?.doubleValue)
        }
    }
    func testFiveObjectivesAreFindableWithSpecificCriteriaAndNoCodeInstructions()throws {
        let item=try XCTUnwrap(LearnCatalogContent.historicalItems.first {$0.id == "optimizer-updates"})
        XCTAssertEqual(item.category,"Training & evaluation")
        for task in OptimizerLearningTask.allCases {
            XCTAssertTrue(item.matches(section:"Practice",search:task.title))
            let guide=LessonAudienceGuide.make(operation:task.rawValue);XCTAssertEqual(guide.goal,task.title);XCTAssertEqual(guide.steps.count,3)
            XCTAssertTrue(guide.steps[0].contains("optimizer"));XCTAssertTrue(guide.steps[1].contains("Learning rate"))
            let requirements=LessonAudienceGuide.requirements(operation:task.rawValue,fallback:"")
            for header in ["**Build**","**Configure**","**Check**"] {XCTAssertTrue(requirements.contains(header))}
            let plan=try OptimizerAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:41)
            for criterion in plan.criteria {XCTAssertNotNil(LessonAudienceGuide.criterionFeedback(operation:task.rawValue,criterionID:criterion.id))}
        }
    }
    func testFiveCurrentTrainOptimizerPracticesAreUniqueSearchableAndInline()throws {
        let entries=LearnCatalogContent.displayedComponentExampleEntries.filter { OptimizerConfiguration.componentIDs.contains($0.componentID) }
        XCTAssertEqual(entries.count,5)
        XCTAssertEqual(Set(entries.map(\.componentID)),Set(OptimizerConfiguration.componentIDs))
        for task in OptimizerLearningTask.allCases {
            let entry=try XCTUnwrap(entries.first {$0.componentID == task.componentID})
            XCTAssertEqual(entry.item.category,"Training & evaluation")
            XCTAssertEqual(entry.item.kind,"Example")
            XCTAssertTrue(entry.item.matches(section:"Examples",search:task.componentTitle))
            XCTAssertTrue(entry.item.matches(section:"Examples",search:task.componentID))
            let document=try PreparedWorkflowExample.component(entry.componentID).document(device:"cpu")
            XCTAssertFalse(document.nodes.contains {if case .optimizerRecipe = $0.payload {return true};return false})
            let train=try XCTUnwrap(document.nodes.first {if case .train = $0.payload {return true};return false})
            if case .train(let config)=train.payload {
                let optimizer=try OptimizerConfiguration.decodeRecipe(try XCTUnwrap(config["optimizer"]))
                XCTAssertEqual(optimizer.componentID,task.componentID)
            }
            XCTAssertFalse(document.edges.contains {$0.kind == .optimizerRecipe})
        }
    }

}
