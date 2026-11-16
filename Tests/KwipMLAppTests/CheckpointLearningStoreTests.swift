import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class CheckpointLearningStoreTests: XCTestCase {
    private func fixture() -> (WorkspaceStore, TensorLearningStore) {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        return (workspace, TensorLearningStore(workspace: workspace))
    }
    func testBothCheckpointExercisesOpenAsManagedLessonsWithoutInventedForwardCanvas() throws {
        for operation in ["checkpoint-restore-weights", "checkpoint-continue-training"] {
            let (workspace, lesson) = fixture()
            guard lesson.start(operation: operation) else { XCTFail("Checkpoint exercise must open: \(operation)"); continue }
            XCTAssertTrue(lesson.isActive)
            XCTAssertEqual(lesson.draft?.operation, operation)
            XCTAssertNil(lesson.canvas, "Artifact controls must not masquerade as a forward graph")
            XCTAssertTrue(lesson.draft?.graph.nodes.isEmpty == true)
            XCTAssertNotNil(lesson.publicTask)
            XCTAssertFalse(lesson.canSubmit, "Undiscovered lifecycle support must not execute")
            XCTAssertTrue(lesson.runtimeReadinessError?.localizedCaseInsensitiveContains("checkpoint") == true)
            XCTAssertTrue(lesson.canRefreshLessonRuntime)
            XCTAssertTrue(lesson.save())
            let projectID = try XCTUnwrap(workspace.activeProjectID)
            let document = workspace.graph
            XCTAssertTrue(workspace.beginProject(GraphDocument(name: "Another project")))
            XCTAssertFalse(lesson.isActive)
            XCTAssertTrue(workspace.beginProject(document, id: projectID))
            XCTAssertTrue(lesson.isActive)
            XCTAssertEqual(lesson.draft?.operation, operation)
            XCTAssertNil(lesson.canvas)
            XCTAssertTrue(lesson.setMode(.assessment))
            XCTAssertFalse(lesson.resultIsCurrent)
        }
    }
    func testControlsPreserveUnappliedEditsAcrossSaveAndReopenAndRefuseInvalidBounds() throws {
        let (workspace,lesson)=fixture()
        XCTAssertTrue(lesson.start(operation:"checkpoint-restore-weights"))
        workspace.lifecycleRuntimeCapability = LifecycleRuntimeCapability(descriptor:try JSONEncoder().encode(BundledLifecycleCapabilities.entry.runtimeContract))
        XCTAssertTrue(lesson.canSubmit,"Fresh controls can execute; grading decides whether they fulfill the task")
        XCTAssertTrue(lesson.setCheckpointField("trainingMode",text:"fineTune"))
        XCTAssertFalse(lesson.canSubmit)
        XCTAssertTrue(lesson.applyCheckpointFields(),"Incomplete parent selection remains editable")
        XCTAssertFalse(lesson.canSubmit,"Restore needs a supplied parent before submission")
        XCTAssertTrue(lesson.setCheckpointField("parent",text:"savedLessonRun"))
        XCTAssertTrue(lesson.applyCheckpointFields())
        XCTAssertTrue(lesson.canSubmit)
        XCTAssertTrue(lesson.setCheckpointField("epochs",text:"9"))
        XCTAssertFalse(lesson.applyCheckpointFields())
        XCTAssertEqual(lesson.checkpointField("epochs"),"9")
        XCTAssertEqual(lesson.checkpointConfiguration?.epochs,1)
        XCTAssertFalse(lesson.canSubmit)
        XCTAssertTrue(lesson.save())
        let id=try XCTUnwrap(workspace.activeProjectID),document=workspace.graph
        XCTAssertTrue(workspace.beginProject(GraphDocument(name:"Another project")))
        XCTAssertTrue(workspace.beginProject(document,id:id))
        XCTAssertEqual(lesson.checkpointField("epochs"),"9")
        XCTAssertFalse(lesson.canSubmit)
        XCTAssertTrue(lesson.discardCheckpointFields())
        XCTAssertEqual(lesson.checkpointField("epochs"),"1")
        XCTAssertTrue(lesson.canSubmit)
        XCTAssertFalse(lesson.setCheckpointField("unreviewed",text:"value"))
        XCTAssertFalse(lesson.setCheckpointField("epochs",text:String(repeating:"1",count:513)))
        XCTAssertTrue(workspace.beginProject(GraphDocument(name:"Inactive")))
        XCTAssertFalse(lesson.setCheckpointField("epochs",text:"2"))
        XCTAssertFalse(lesson.applyCheckpointFields())
    }
    func testCatalogMakesBothExercisesFindableWithoutForwardComponentIdentity() throws {
        let item=try XCTUnwrap(LearnCatalogContent.items.first {$0.id == "checkpoint-lifecycle"})
        XCTAssertEqual(item.category,"Training & evaluation")
        XCTAssertEqual(item.kind,"Construction")
        for task in CheckpointLearningTask.allCases {
            XCTAssertTrue(item.matches(section:"Library",search:task.rawValue))
            XCTAssertTrue(item.matches(section:"Library",search:task.title))
        }
        for term in ["save","load","fine-tune","resume","weights"] {XCTAssertTrue(item.matches(section:"Library",search:term))}
        XCTAssertNil(try ComponentRegistry.builtIn().component(type:"checkpoint_lifecycle"))
    }

    func testWorkedExamplesPrepareRealTaskControlsAndKeepDestinationWindowOwnership() throws {
        XCTAssertEqual(CheckpointLearningExamples.all.map(\.id),["prepared-checkpoint-restore-weights","prepared-checkpoint-continue-training"])
        let item=try XCTUnwrap(LearnCatalogContent.items.first {$0.id == "checkpoint-examples"})
        XCTAssertEqual(item.kind,"Example")
        for example in CheckpointLearningExamples.all {
            let (workspace,lesson)=fixture()
            XCTAssertTrue(lesson.start(operation:example.id))
            XCTAssertNotNil(workspace.activeProjectID)
            XCTAssertEqual(lesson.draft?.operation,example.task.rawValue)
            XCTAssertEqual(lesson.checkpointConfiguration,example.configuration)
            XCTAssertEqual(lesson.draft?.mode,.guided)
            XCTAssertTrue(lesson.draft?.pendingFields.isEmpty == true)
            XCTAssertNil(lesson.canvas)
            XCTAssertTrue(item.matches(section:"Examples",search:example.id))
            XCTAssertFalse(item.matches(section:"Practice",search:example.id))
            var destination:String?
            lesson.startInWindow={operation,_ in destination=operation;return true}
            let before=lesson.draft
            XCTAssertTrue(lesson.start(operation:example.id))
            XCTAssertEqual(destination,example.id)
            XCTAssertEqual(lesson.draft,before,"Only the receiving window may prepare example controls")
        }
    }

}
