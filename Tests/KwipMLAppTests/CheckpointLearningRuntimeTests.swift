import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class CheckpointLearningRuntimeTests:XCTestCase {
    private func apply(_ configuration:CheckpointLearningConfiguration,to lesson:TensorLearningStore) {
        XCTAssertTrue(lesson.setCheckpointField("trainingMode",text:configuration.trainingMode.rawValue))
        XCTAssertTrue(lesson.setCheckpointField("parent",text:configuration.parent.rawValue))
        XCTAssertTrue(lesson.setCheckpointField("epochs",text:String(configuration.epochs)))
        XCTAssertTrue(lesson.setCheckpointField("batchSize",text:String(configuration.batchSize)))
        XCTAssertTrue(lesson.applyCheckpointFields(),lesson.statusMessage)
    }
    private func wait(_ workspace:WorkspaceStore) async throws {
        let deadline=Date().addingTimeInterval(65)
        while workspace.isAssessingLearning && Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
        XCTAssertFalse(workspace.isAssessingLearning,"Actual worker must finish inside the existing lesson budget")
    }
    func testBothActualWorkerLessonsAllModesAndColdReceiptReopen() async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        XCTAssertTrue(FileManager.default.fileExists(atPath:repository.appendingPathComponent(".venv/bin/python").path),"Actual development runtime required")
        for task in CheckpointLearningTask.allCases {
            let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer {try? FileManager.default.removeItem(at:root)}
            let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
            let lesson=TensorLearningStore(workspace:workspace)
            XCTAssertTrue(lesson.start(operation:task.rawValue))
            await verifyRuntimeForIntegration(workspace)
            XCTAssertNil(lesson.runtimeReadinessError,workspace.capabilityStatus)
            apply(task.configuration,to:lesson)
            for mode in [LearningMode.guided,.practice,.assessment] {
                XCTAssertTrue(lesson.setMode(mode))
                XCTAssertTrue(lesson.canSubmit,lesson.statusMessage)
                lesson.submit();try await wait(workspace)
                XCTAssertEqual(lesson.grade?.status,.passed,lesson.statusMessage)
                XCTAssertTrue(lesson.resultSaved,lesson.statusMessage)
                XCTAssertTrue(lesson.resultIsCurrent)
                if mode == .assessment {XCTAssertNil(lesson.checkpointProjection)}
                else {XCTAssertEqual(lesson.checkpointProjection?.fixtures.count,4)}
                let projectID=try XCTUnwrap(workspace.activeProjectID),document=workspace.graph
                XCTAssertTrue(workspace.beginProject(GraphDocument(name:"Another project")))
                XCTAssertTrue(workspace.beginProject(document,id:projectID))
                XCTAssertTrue(lesson.resultSaved,lesson.statusMessage)
                XCTAssertEqual(lesson.grade?.status,.passed,lesson.statusMessage)
                XCTAssertTrue(lesson.resultIsCurrent)
                XCTAssertTrue(lesson.statusMessage.contains(mode == .assessment ? "assessment verified":"example check verified"),lesson.statusMessage)
                XCTAssertNil(lesson.canvas)
                if mode == .assessment {XCTAssertNil(lesson.checkpointProjection)}
                else {XCTAssertEqual(lesson.checkpointProjection?.fixtures.count,4)}
                let coldWorkspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
                let coldLesson=TensorLearningStore(workspace:coldWorkspace)
                XCTAssertTrue(coldWorkspace.beginProject(document,id:projectID))
                XCTAssertTrue(coldLesson.isActive)
                XCTAssertEqual(coldLesson.grade?.status,.passed,coldLesson.statusMessage)
                XCTAssertTrue(coldLesson.resultSaved)
                XCTAssertTrue(coldLesson.resultIsCurrent)
                XCTAssertFalse(coldWorkspace.isAssessingLearning,"Receipt replay must not rerun training")
                if mode == .assessment {XCTAssertNil(coldLesson.checkpointProjection)}
                else {XCTAssertEqual(coldLesson.checkpointProjection?.fixtures.count,4)}
            }
            XCTAssertTrue(lesson.setMode(.practice))
            XCTAssertTrue(lesson.setCheckpointField("epochs",text:"3"))
            XCTAssertNil(lesson.checkpointProjection)
            XCTAssertFalse(lesson.canSubmit)
            XCTAssertTrue(lesson.discardCheckpointFields())
            if task == .continueTraining {
                apply(try .init(trainingMode:.fresh,parent:.none,epochs:4,batchSize:4),to:lesson)
                lesson.submit();try await wait(workspace)
                XCTAssertEqual(lesson.grade?.status,.failed,lesson.statusMessage)
                XCTAssertTrue(lesson.resultSaved)
                XCTAssertTrue(lesson.grade?.outcomes.contains {$0.criterionID == "continuationEpochs" && !$0.passed} == true,
                    "Matching final numbers must not pass fresh training as continuation")
            }
        }
    }
}
