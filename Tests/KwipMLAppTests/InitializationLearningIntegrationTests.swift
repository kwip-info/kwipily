import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class InitializationLearningIntegrationTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    func testBothHistoricalLessonsRetainTheirCanvasButCannotRefreshOrSubmit()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let lesson=TensorLearningStore(workspace:workspace)
        for operation in ["initialization-starting-values","initialization-starting-spread"] {
            XCTAssertTrue(lesson.start(operation:operation),lesson.statusMessage)
            guard lesson.isActive else {continue}
            let canvas=try XCTUnwrap(lesson.canvas)
            XCTAssertEqual(Set(canvas.registry.components.map(\.id)),["initialization","optimizer_sgd","train"])
            XCTAssertEqual(canvas.graph.schemaVersion,9)
            XCTAssertFalse(canvas.canAddType("input"));XCTAssertFalse(canvas.canAddType("linear"))
            XCTAssertNotNil(lesson.runtimeReadinessError)
            XCTAssertFalse(lesson.canSubmit)
            XCTAssertFalse(lesson.canRefreshLessonRuntime)
            XCTAssertTrue(lesson.runtimeReadinessError?.contains("read-only") == true)
        }
    }
    func testStartingSettingsGoalsAreFindableAndExplainEveryMeasuredCriterion()throws {
        let item=try XCTUnwrap(LearnCatalogContent.historicalItems.first {$0.id == "initialization-starting-settings"})
        XCTAssertEqual(item.category,"Training & evaluation")
        for task in InitializationLearningTask.allCases {
            XCTAssertTrue(item.matches(section:"Practice",search:task.title))
            let guide=LessonAudienceGuide.make(operation:task.rawValue)
            XCTAssertEqual(guide.goal,task.title)
            XCTAssertEqual(guide.steps.count,3)
            XCTAssertTrue(guide.steps.joined(separator:" ").contains("Starting settings"))
            let requirements=LessonAudienceGuide.requirements(operation:task.rawValue,fallback:"")
            for header in ["**Build**","**Configure**","**Check**"] {XCTAssertTrue(requirements.contains(header))}
            for key in ["startingSettings","predictions","firstAdjustment","repeatIsolation","savedReuse"] {
                let feedback=try XCTUnwrap(LessonAudienceGuide.criterionFeedback(operation:task.rawValue,criterionID:key))
                XCTAssertFalse(feedback.title == "Construction check")
                XCTAssertFalse(feedback.nextStep.isEmpty)
            }
        }
    }
    func testTypedLessonOwnerDraftPersistsWithoutEditingSuppliedContextAndSupportsUndo()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false),lesson=TensorLearningStore(workspace:workspace)
        XCTAssertTrue(lesson.start(operation:"initialization-starting-values"))
        let canvas=try XCTUnwrap(lesson.canvas),node=try XCTUnwrap(canvas.graph.nodes.first {$0.type == "initialization"}),train=try XCTUnwrap(canvas.graph.nodes.first {$0.type == "train"})
        XCTAssertEqual(canvas.graph.edges.count,0,"Opening a lesson must not connect or solve it automatically")
        XCTAssertFalse(canvas.configure(nodeID:train.id,values:["epochs":2]))
        canvas.selectNode(node.id,extending:false)
        let context=try XCTUnwrap(canvas.initializationInspector(nodeID:node.id))
        XCTAssertEqual(Set(context.owners.map {$0.node+"."+$0.parameter}),["first.weight","first.bias","second.weight","second.bias"])
        XCTAssertEqual(context.owners.first {$0.node == "first" && $0.parameter == "weight"}?.shape,[3,2])
        XCTAssertTrue(context.setSeedText("not a seed"));XCTAssertFalse(context.canApply)
        XCTAssertFalse(try XCTUnwrap(lesson.draft).pendingFields.isEmpty)
        XCTAssertThrowsError(try lesson.draft?.submissionData())
        XCTAssertTrue(lesson.drafts.save())
        let id=try XCTUnwrap(workspace.activeProjectID),document=workspace.graph
        let cold=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false),reopened=TensorLearningStore(workspace:cold)
        XCTAssertTrue(cold.beginProject(document,id:id))
        let restoredCanvas=try XCTUnwrap(reopened.canvas);restoredCanvas.selectNode(node.id,extending:false)
        let restored=try XCTUnwrap(restoredCanvas.initializationInspector(nodeID:node.id))
        XCTAssertEqual(restored.seedText,"not a seed");XCTAssertFalse(restored.canApply)
        XCTAssertTrue(restored.discard());XCTAssertTrue(try XCTUnwrap(reopened.draft).pendingFields.isEmpty)
        XCTAssertTrue(context.discard());XCTAssertTrue(context.setSeedText("43"));XCTAssertTrue(context.apply())
        XCTAssertEqual(canvas.graph.nodes.first {$0.id == node.id}?.config["seed"]?.integerValue,43)
        canvas.undo();XCTAssertEqual(canvas.graph.nodes.first {$0.id == node.id}?.config["seed"]?.integerValue,42)
        canvas.selectNode(train.id,extending:false);canvas.selectNode(node.id,extending:false)
        XCTAssertFalse(context.setSeedText("44"),"Departed selected-node callbacks must not revive")
        XCTAssertEqual(canvas.graph.schemaVersion,9)
    }

}
