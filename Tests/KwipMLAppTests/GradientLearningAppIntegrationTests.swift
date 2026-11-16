import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class GradientLearningAppIntegrationTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    func testTypedCommandsPendingUndoColdPersistenceBusyAndStaleRefusal()throws {
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let w=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),lesson=TensorLearningStore(workspace:w)
        XCTAssertTrue(lesson.start(operation:GradientLearningTask.combineSmallGroups.rawValue))
        let canvas=try XCTUnwrap(lesson.canvas),correct=try GradientPolicyConfiguration(values:GradientLearningTask.combineSmallGroups.gradientValues)
        XCTAssertFalse(canvas.configure(nodeID:"optimizer",values:["lr":1]));XCTAssertFalse(canvas.configure(nodeID:"train",values:["seed":7]))
        XCTAssertFalse(canvas.configureGradientPolicy(nodeID:"trainability",configuration:correct))
        XCTAssertTrue(canvas.configureGradientPolicy(nodeID:"gradient",configuration:correct))
        XCTAssertEqual(canvas.graph.nodes.first {$0.id == "gradient"}?.config["accumulation_steps"],2)
        canvas.undo();XCTAssertEqual(canvas.graph.nodes.first {$0.id == "gradient"}?.config["accumulation_steps"],1)
        canvas.redo();XCTAssertEqual(canvas.graph.nodes.first {$0.id == "gradient"}?.config["accumulation_steps"],2)
        canvas.assessmentIsRunning={true};XCTAssertFalse(canvas.configureGradientPolicy(nodeID:"gradient",configuration:try GradientPolicyConfiguration()))
        canvas.assessmentIsRunning={false}
        XCTAssertTrue(canvas.setPending(nodeID:"gradient",field:"accumulation_steps",text:"-"))
        XCTAssertThrowsError(try lesson.draft?.submissionData());XCTAssertTrue(lesson.drafts.save())
        let id=try XCTUnwrap(w.activeProjectID),document=w.graph
        let cold=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),reopened=TensorLearningStore(workspace:cold)
        XCTAssertTrue(cold.beginProject(document,id:id))
        XCTAssertEqual(reopened.canvas?.pendingText(nodeID:"gradient",field:"accumulation_steps"),"-")
        reopened.canvas?.discardPending(nodeID:"gradient",field:"accumulation_steps")
        XCTAssertTrue(reopened.draft?.pendingFields.isEmpty == true)
        XCTAssertTrue(lesson.start(operation:GradientLearningTask.keepSettings.rawValue))
        XCTAssertFalse(canvas.configureGradientPolicy(nodeID:"gradient",configuration:correct))
        let active=try XCTUnwrap(lesson.canvas),flags=try TrainabilityConfiguration(values:GradientLearningTask.keepSettings.trainabilityValues)
        XCTAssertTrue(active.configureTrainability(nodeID:"trainability",configuration:flags))
        XCTAssertFalse(active.configureTrainability(nodeID:"gradient",configuration:flags))
        active.undo();XCTAssertEqual(active.graph.nodes.first {$0.id == "trainability"}?.config["owners"],.array([]))
    }
    func testFourPreparedLessonExamplesResolveWithPublishedGeneralPalette()throws {
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let w=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),lesson=TensorLearningStore(workspace:w)
        for id in ["gradient_policy","trainability"] {
            XCTAssertEqual(w.registry.component(type:id),BundledGradientControlComponents.entries[id]?.definition)
        }
        for task in GradientLearningTask.allCases {
            XCTAssertTrue(lesson.start(operation:"prepared-"+task.rawValue),lesson.statusMessage)
            XCTAssertEqual(lesson.draft?.operation,task.rawValue)
            XCTAssertEqual(lesson.canvas?.graph,try GradientLearningExamples.preparedGraph(task:task))
            XCTAssertThrowsError(try lesson.draft?.submission())
        }
    }
}
