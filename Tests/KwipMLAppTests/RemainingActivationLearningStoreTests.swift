import XCTest
import KwipMLCore
@testable import KwipMLApp

final class RemainingActivationLearningStoreTests:XCTestCase {
    @MainActor func testEveryTaskWorkerColdReceiptAndRepresentativeModes()async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
        _ = try XCTUnwrap(workspace.runtime)
        let lesson=TensorLearningStore(workspace:workspace)
        await verifyRuntimeForIntegration(workspace)
        func wait()async throws {
            let deadline=Date().addingTimeInterval(30)
            while workspace.isAssessingLearning,Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
            XCTAssertFalse(workspace.isAssessingLearning)
        }
        for task in RemainingActivationLearningTask.allCases {
            XCTAssertTrue(lesson.start(operation:task.rawValue),lesson.statusMessage)
            XCTAssertNotNil(lesson.publicTask)
            let graph=GraphDocument(name:"My measurements",nodes:[.init(id:"source",type:"input",x:0,y:0),.init(id:"response",type:task.componentID,x:200,y:0,config:task.configuration),.init(id:"result",type:"output",x:400,y:0)],edges:[.init(id:"first",source:"source",target:"response"),.init(id:"second",source:"response",target:"result")])
            XCTAssertTrue(lesson.drafts.update {
                $0.graph=graph;$0.parameters=task.slopes.map {["response":["weight":$0]]}
                $0.gradientNodeID=task.isPReLU ? "response":nil
            })
            // All tasks retain real numerical workers; one task covers all mode lifecycles.
            let workerModes:[LearningMode] = task == .sharedSlope ? [.guided,.practice,.assessment]:[.guided]
            for mode in workerModes {
                XCTAssertTrue(lesson.setMode(mode))
                XCTAssertTrue(lesson.canSubmit,lesson.statusMessage)
                lesson.submit();try await wait()
                XCTAssertEqual(lesson.grade?.status,.passed,"\(task.rawValue) \(mode): \(lesson.statusMessage)")
                XCTAssertTrue(lesson.resultSaved)
                XCTAssertEqual(lesson.formativeProjection == nil,mode == .assessment)
            }
            let id=try XCTUnwrap(workspace.activeProjectID),saved=workspace.graph
            XCTAssertTrue(workspace.beginProject(GraphDocument(name:"Separate work")))
            XCTAssertTrue(workspace.beginProject(saved,id:id))
            XCTAssertEqual(lesson.grade?.status,.passed)
            XCTAssertTrue(lesson.resultSaved);XCTAssertEqual(lesson.formativeProjection == nil,task == .sharedSlope)
            let coldWorkspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
            let coldLesson=TensorLearningStore(workspace:coldWorkspace)
            await verifyRuntimeForIntegration(coldWorkspace)
            XCTAssertTrue(coldWorkspace.beginProject(saved,id:id))
            XCTAssertEqual(coldLesson.grade?.status,.passed)
            XCTAssertTrue(coldLesson.resultSaved);XCTAssertEqual(coldLesson.formativeProjection == nil,task == .sharedSlope)
            let savedDrafts=TensorLearningDraftRepository(directory:storage.appendingPathComponent("learning/projects"))
            for mode in [LearningMode.practice,.assessment,.guided] {
                XCTAssertTrue(lesson.setMode(mode));XCTAssertTrue(lesson.save())
                XCTAssertNil(lesson.grade);XCTAssertNil(lesson.formativeProjection);XCTAssertFalse(lesson.resultSaved)
                XCTAssertEqual(try savedDrafts.load(projectID:id).draft.mode,mode)
                if task == .sharedSlope {
                    let reopened=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
                    let reopenedLesson=TensorLearningStore(workspace:reopened)
                    XCTAssertTrue(reopened.beginProject(saved,id:id));XCTAssertEqual(reopenedLesson.draft?.mode,mode)
                    if mode == .assessment {XCTAssertNil(reopenedLesson.formativeProjection)}
                }
            }
        }
    }
}
