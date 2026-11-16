import XCTest
import KwipMLCore
@testable import KwipMLApp

final class SpatialPatchLearningStoreTests:XCTestCase {
    private func construction(_ task:SpatialPatchLearningTask)->GraphDocument {
        .init(name:"My patch measurements",nodes:[.init(id:"source",type:"input",name:"My grid",x:0,y:0),.init(id:"patch",type:task.componentID,name:"My arrangement",x:200,y:0,config:task.configuration),.init(id:"result",type:"output",name:"My result",x:400,y:0)],edges:[.init(id:"first",source:"source",target:"patch"),.init(id:"second",source:"patch",target:"result")])
    }
    @MainActor func testEveryTaskWorkerColdReceiptAndRepresentativeModes()async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),lesson=TensorLearningStore(workspace:workspace)
        _ = try XCTUnwrap(workspace.runtime)
        XCTAssertTrue(lesson.start(operation:SpatialPatchLearningTask.unfoldOverlap.rawValue));XCTAssertNotNil(lesson.runtimeReadinessError,"New family support must default absent")
        await verifyRuntimeForIntegration(workspace)
        func wait()async throws {let until=Date().addingTimeInterval(30);while workspace.isAssessingLearning,Date()<until {try await Task.sleep(for:.milliseconds(20))};XCTAssertFalse(workspace.isAssessingLearning)}
        for task in SpatialPatchLearningTask.allCases {
            XCTAssertTrue(lesson.start(operation:task.rawValue),lesson.statusMessage);XCTAssertNil(lesson.runtimeReadinessError)
            XCTAssertTrue(lesson.drafts.update {$0.graph=construction(task)})
            XCTAssertNotNil(lesson.publicTask)
            // All tasks retain real numerical workers; one task covers all mode lifecycles.
            let workerModes:[LearningMode] = task == .unfoldOverlap ? [.guided,.practice,.assessment]:[.guided]
            for mode in workerModes {
                XCTAssertTrue(lesson.setMode(mode));XCTAssertTrue(lesson.canSubmit,lesson.statusMessage)
                lesson.submit();try await wait();XCTAssertEqual(lesson.grade?.status,.passed,"\(task.rawValue) \(mode): \(lesson.statusMessage)")
                XCTAssertTrue(lesson.resultSaved);XCTAssertEqual(lesson.formativeProjection == nil,mode == .assessment)
            }
            let id=try XCTUnwrap(workspace.activeProjectID),graph=workspace.graph
            let cold=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),coldLesson=TensorLearningStore(workspace:cold)
            await verifyRuntimeForIntegration(cold);XCTAssertTrue(cold.beginProject(graph,id:id))
            XCTAssertEqual(coldLesson.grade?.status,.passed);XCTAssertTrue(coldLesson.resultSaved);XCTAssertEqual(coldLesson.formativeProjection == nil,task == .unfoldOverlap)
            let savedDrafts=TensorLearningDraftRepository(directory:storage.appendingPathComponent("learning/projects"))
            for mode in [LearningMode.practice,.assessment,.guided] {
                XCTAssertTrue(lesson.setMode(mode));XCTAssertTrue(lesson.save())
                XCTAssertNil(lesson.grade);XCTAssertNil(lesson.formativeProjection);XCTAssertFalse(lesson.resultSaved)
                XCTAssertEqual(try savedDrafts.load(projectID:id).draft.mode,mode)
                if task == .unfoldOverlap {
                    let reopened=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
                    let reopenedLesson=TensorLearningStore(workspace:reopened)
                    XCTAssertTrue(reopened.beginProject(graph,id:id));XCTAssertEqual(reopenedLesson.draft?.mode,mode)
                    if mode == .assessment {XCTAssertNil(reopenedLesson.formativeProjection)}
                }
            }
        }
    }
    @MainActor func testValidSameShapeWrongSpacingIsMeasuredAndRepairPasses()async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),lesson=TensorLearningStore(workspace:workspace)
        await verifyRuntimeForIntegration(workspace);XCTAssertTrue(lesson.start(operation:SpatialPatchLearningTask.unfoldSpaced.rawValue))
        var wrong=construction(.unfoldSpaced);wrong.nodes[1].config=SpatialPatchLearningTask.unfoldOverlap.configuration
        // Both arrangements produce [N,8,6]; only actual values/influence distinguish them.
        XCTAssertEqual(SpatialPatchLearningTask.unfoldSpaced.outputSampleShape,SpatialPatchLearningTask.unfoldOverlap.outputSampleShape)
        XCTAssertTrue(lesson.drafts.update {$0.graph=wrong});XCTAssertTrue(lesson.canSubmit,lesson.statusMessage)
        func wait()async throws {let until=Date().addingTimeInterval(30);while workspace.isAssessingLearning,Date()<until {try await Task.sleep(for:.milliseconds(20))};XCTAssertFalse(workspace.isAssessingLearning)}
        lesson.submit();try await wait();XCTAssertEqual(lesson.grade?.status,.failed);XCTAssertNotNil(lesson.formativeProjection);XCTAssertTrue(lesson.resultSaved)
        XCTAssertTrue(lesson.drafts.update {$0.graph=construction(.unfoldSpaced)});XCTAssertFalse(lesson.resultIsCurrent)
        lesson.submit();try await wait();XCTAssertEqual(lesson.grade?.status,.passed);XCTAssertTrue(lesson.resultSaved)
    }
}
