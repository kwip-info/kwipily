import XCTest
import KwipMLCore
@testable import KwipMLApp
final class TensorRoutingLearningStoreTests:XCTestCase {
    @MainActor func testEveryTaskWorkerColdReceiptAndRepresentativeModes()async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),lesson=TensorLearningStore(workspace:workspace)
        _ = try XCTUnwrap(workspace.runtime)
        XCTAssertTrue(lesson.start(operation:TensorRoutingLearningTask.chunk.rawValue));XCTAssertNotNil(lesson.runtimeReadinessError,"New family support defaults absent")
        await verifyRuntimeForIntegration(workspace)
        func wait()async throws {let until=Date().addingTimeInterval(30);while workspace.isAssessingLearning,Date()<until {try await Task.sleep(for:.milliseconds(20))};XCTAssertFalse(workspace.isAssessingLearning)}
        for task in TensorRoutingLearningTask.allCases {
            XCTAssertTrue(lesson.start(operation:task.rawValue));XCTAssertNil(lesson.runtimeReadinessError)
            XCTAssertTrue(lesson.drafts.update {$0.graph=routingLessonGraph(task)})
            XCTAssertNotNil(lesson.publicTask)
            // All tasks retain real numerical workers; one task covers all mode lifecycles.
            let workerModes:[LearningMode] = task == .chunk ? [.guided,.practice,.assessment]:[.guided]
            for mode in workerModes {
                XCTAssertTrue(lesson.setMode(mode));XCTAssertTrue(lesson.canSubmit,lesson.statusMessage)
                lesson.submit();try await wait();XCTAssertEqual(lesson.grade?.status,.passed,"\(task.rawValue) \(mode): \(lesson.statusMessage)")
                XCTAssertTrue(lesson.resultSaved);XCTAssertEqual(lesson.formativeProjection == nil,mode == .assessment)
            }
            let id=try XCTUnwrap(workspace.activeProjectID),graph=workspace.graph
            let cold=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),coldLesson=TensorLearningStore(workspace:cold)
            await verifyRuntimeForIntegration(cold);XCTAssertTrue(cold.beginProject(graph,id:id))
            XCTAssertEqual(coldLesson.grade?.status,.passed);XCTAssertTrue(coldLesson.resultSaved);XCTAssertEqual(coldLesson.formativeProjection == nil,task == .chunk)
            let savedDrafts=TensorLearningDraftRepository(directory:storage.appendingPathComponent("learning/projects"))
            for mode in [LearningMode.practice,.assessment,.guided] {
                XCTAssertTrue(lesson.setMode(mode));XCTAssertTrue(lesson.save())
                XCTAssertNil(lesson.grade);XCTAssertNil(lesson.formativeProjection);XCTAssertFalse(lesson.resultSaved)
                XCTAssertEqual(try savedDrafts.load(projectID:id).draft.mode,mode)
                if task == .chunk {
                    let reopened=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
                    let reopenedLesson=TensorLearningStore(workspace:reopened)
                    XCTAssertTrue(reopened.beginProject(graph,id:id));XCTAssertEqual(reopenedLesson.draft?.mode,mode)
                    if mode == .assessment {XCTAssertNil(reopenedLesson.formativeProjection)}
                }
            }
        }
    }
    @MainActor func testValidWrongRejoinIsMeasuredAndRepairPasses()async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),lesson=TensorLearningStore(workspace:workspace)
        await verifyRuntimeForIntegration(workspace);XCTAssertTrue(lesson.start(operation:TensorRoutingLearningTask.splitSize.rawValue))
        var wrong=routingLessonGraph(.splitSize)
        for index in wrong.edges.indices where wrong.edges[index].source == "op" {wrong.edges[index].sourcePort=["part2":"part0","part0":"part1","part1":"part2"][wrong.edges[index].sourcePort]!}
        XCTAssertTrue(lesson.drafts.update {$0.graph=wrong});XCTAssertTrue(lesson.canSubmit)
        func wait()async throws {let until=Date().addingTimeInterval(30);while workspace.isAssessingLearning,Date()<until {try await Task.sleep(for:.milliseconds(20))};XCTAssertFalse(workspace.isAssessingLearning)}
        lesson.submit();try await wait();XCTAssertEqual(lesson.grade?.status,.failed);XCTAssertNotNil(lesson.formativeProjection)
        XCTAssertTrue(lesson.drafts.update {$0.graph=routingLessonGraph(.splitSize)});XCTAssertFalse(lesson.resultIsCurrent)
        lesson.submit();try await wait();XCTAssertEqual(lesson.grade?.status,.passed)
    }
}
