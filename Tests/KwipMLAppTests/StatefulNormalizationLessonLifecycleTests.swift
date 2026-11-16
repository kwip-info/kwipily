import XCTest
import KwipMLCore
@testable import KwipMLApp

final class StatefulNormalizationLessonLifecycleTests:XCTestCase {
    private func construction(_ task:StatefulNormalizationLearningTask)->GraphDocument {
        .init(name:"My channel comparisons",nodes:[.init(id:"source",type:"input",name:"My measurements",x:0,y:0),.init(id:"norm",type:task.componentID,name:"My comparison",x:200,y:0,config:task.configuration),.init(id:"result",type:"output",name:"My result",x:400,y:0)],edges:[.init(id:"first",source:"source",target:"norm"),.init(id:"second",source:"norm",target:"result")])
    }
    @MainActor private func configure(_ lesson:TensorLearningStore,_ task:StatefulNormalizationLearningTask,graph:GraphDocument?=nil)->Bool {
        lesson.drafts.update {d in
            d.graph=graph ?? construction(task)
            if let scale=task.scale,let bias=task.bias {d.parameters=["norm":["weight":scale,"bias":bias]];d.gradientNodeID="norm"}
            else {d.parameters=nil;d.gradientNodeID=nil}
        }
    }
    @MainActor private func wait(_ workspace:WorkspaceStore)async throws {
        let until=Date().addingTimeInterval(30)
        while workspace.isAssessingLearning,Date()<until {try await Task.sleep(for:.milliseconds(20))}
        XCTAssertFalse(workspace.isAssessingLearning)
    }
    @MainActor func testEveryScientificTaskWorkerColdReceiptAndRepresentativeThreeModeLifecycle()async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),lesson=TensorLearningStore(workspace:workspace)
        _ = try XCTUnwrap(workspace.runtime)
        XCTAssertTrue(lesson.start(operation:StatefulNormalizationLearningTask.bn1Current.rawValue));XCTAssertNotNil(lesson.runtimeReadinessError,"New descriptors must default absent")
        await verifyRuntimeForIntegration(workspace)
        for task in StatefulNormalizationLearningTask.allCases {
            XCTAssertTrue(lesson.start(operation:task.rawValue),lesson.statusMessage);XCTAssertNil(lesson.runtimeReadinessError)
            XCTAssertTrue(configure(lesson,task));XCTAssertNotNil(lesson.publicTask)
            // Numerical dispatch is independent of learning mode. Exercise all modes
            // through one real lifecycle; every scientific task still runs its worker.
            let workerModes:[LearningMode] = task == .bn1Current ? [.guided,.practice,.assessment]:[.guided]
            for mode in workerModes {
                XCTAssertTrue(lesson.setMode(mode));XCTAssertTrue(lesson.canSubmit,lesson.statusMessage)
                lesson.submit();try await wait(workspace)
                XCTAssertEqual(lesson.grade?.status,.passed,"\(task.rawValue) \(mode): \(lesson.statusMessage)")
                XCTAssertTrue(lesson.resultSaved);XCTAssertEqual(lesson.formativeProjection==nil,mode == .assessment)
            }
            let id=try XCTUnwrap(workspace.activeProjectID),graph=workspace.graph
            let cold=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),coldLesson=TensorLearningStore(workspace:cold)
            await verifyRuntimeForIntegration(cold);XCTAssertTrue(cold.beginProject(graph,id:id))
            XCTAssertEqual(coldLesson.grade?.status,.passed,"Cold \(task.rawValue): \(coldLesson.statusMessage)");XCTAssertTrue(coldLesson.resultSaved)
            XCTAssertEqual(coldLesson.formativeProjection == nil,task == .bn1Current)
            let saved=TensorLearningDraftRepository(directory:storage.appendingPathComponent("learning/projects"))
            for mode in [LearningMode.practice,.assessment,.guided] {
                XCTAssertTrue(lesson.setMode(mode));XCTAssertTrue(lesson.save())
                XCTAssertNil(lesson.grade);XCTAssertNil(lesson.formativeProjection);XCTAssertFalse(lesson.resultSaved)
                XCTAssertEqual(try saved.load(projectID:id).draft.mode,mode)
                if task == .bn1Current {
                    let reopened=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
                    let reopenedLesson=TensorLearningStore(workspace:reopened)
                    XCTAssertTrue(reopened.beginProject(graph,id:id))
                    XCTAssertEqual(reopenedLesson.draft?.mode,mode)
                    if mode == .assessment {XCTAssertNil(reopenedLesson.formativeProjection)}
                }
            }
        }
    }
    @MainActor func testValidSameShapeWrongSummaryShareAndChannelTeamsAreMeasuredThenRepairPasses()async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),lesson=TensorLearningStore(workspace:workspace)
        await verifyRuntimeForIntegration(workspace)
        for task in [StatefulNormalizationLearningTask.bn1Remembered,.in2Remembered,.groupAdjustments] {
            XCTAssertTrue(lesson.start(operation:task.rawValue))
            var wrong=construction(task)
            if task == .groupAdjustments {wrong.nodes[1].config["num_groups"] = .number(1)} else {wrong.nodes[1].config["momentum"] = .number(0.75)}
            XCTAssertTrue(configure(lesson,task,graph:wrong));XCTAssertTrue(lesson.canSubmit,lesson.statusMessage)
            lesson.submit();try await wait(workspace);XCTAssertEqual(lesson.grade?.status,.failed);XCTAssertNotNil(lesson.formativeProjection);XCTAssertTrue(lesson.resultSaved)
            XCTAssertTrue(configure(lesson,task));XCTAssertFalse(lesson.resultIsCurrent)
            lesson.submit();try await wait(workspace);XCTAssertEqual(lesson.grade?.status,.passed);XCTAssertTrue(lesson.resultSaved)
        }
    }
}
