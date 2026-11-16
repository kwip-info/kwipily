import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DataProgramLessonLifecycleTests:XCTestCase {
    @MainActor private func context()throws->(WorkspaceStore,TensorLearningStore,URL,URL) {
        guard let path=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Enable the actual development runtime for managed lesson lifecycle checks.")}
        let root=URL(fileURLWithPath:path),storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let workspace=WorkspaceStore(rootOverride:root,storageOverride:storage,detectCapabilities:false)
        return(workspace,TensorLearningStore(workspace:workspace),root,storage)
    }
    @MainActor private func check(_ lesson:TensorLearningStore,workspace:WorkspaceStore)async throws {
        XCTAssertTrue(lesson.canSubmit,lesson.runtimeReadinessError ?? lesson.statusMessage)
        lesson.submit();let deadline=Date().addingTimeInterval(90)
        while workspace.isAssessingLearning && Date()<deadline {try await Task.sleep(nanoseconds:20_000_000)}
        XCTAssertFalse(workspace.isAssessingLearning,"Bounded managed lesson did not finish")
        XCTAssertEqual(lesson.grade?.status,.passed,lesson.statusMessage)
        XCTAssertTrue(lesson.resultSaved,lesson.statusMessage);XCTAssertTrue(lesson.resultIsCurrent)
    }
    @MainActor private func reopen(_ records:[(String,String,GraphDocument)],root:URL,storage:URL)throws {
        let workspace=WorkspaceStore(rootOverride:root,storageOverride:storage,detectCapabilities:false)
        let lesson=TensorLearningStore(workspace:workspace)
        for (operation,id,graph) in records {
            XCTAssertTrue(workspace.beginProject(graph,id:id))
            XCTAssertEqual(lesson.draft?.operation,operation);XCTAssertEqual(lesson.draft?.mode,.assessment)
            XCTAssertEqual(lesson.grade?.status,.passed,lesson.statusMessage)
            XCTAssertTrue(lesson.resultSaved);XCTAssertTrue(lesson.resultIsCurrent)
            XCTAssertNil(lesson.recipeProjection)
            XCTAssertNotNil(lesson.runtimeReadinessError,"Cold receipt replay cannot grant runtime discovery")
            XCTAssertTrue(lesson.setMode(.guided));XCTAssertNil(lesson.grade)
            XCTAssertNil(lesson.recipeProjection,"Private-origin measurement remains hidden after a mode switch")
        }
    }
    @MainActor func testTwoActualArtifactActionLessonsGradePublicPrivateAndReplayCold()async throws {
        let (workspace,lesson,root,storage)=try context();defer {try? FileManager.default.removeItem(at:storage)}
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.programExportRuntimeCapability.isAvailable)
        var records:[(String,String,GraphDocument)]=[]
        for task in ProgramExportLearningTask.allCases {
            XCTAssertTrue(lesson.start(operation:task.rawValue,mode:.guided));XCTAssertNil(lesson.canvas)
            let controls=DataProgramLearningControls(lesson:lesson)
            for key in task.values.keys.sorted() {XCTAssertTrue(controls.setField(key,value:task.values[key]!))}
            XCTAssertTrue(controls.apply())
            try await check(lesson,workspace:workspace)
            let projection=try XCTUnwrap(lesson.recipeProjection)
            XCTAssertEqual(projection.fixtures.count,4)
            for fixture in projection.fixtures {
                XCTAssertEqual(fixture.measured["artifactChecks"]?.values,[1,1,1])
                XCTAssertEqual(fixture.measured["constraintChecks"]?.values,[1,1,1,1])
                XCTAssertEqual(fixture.measured["parentByteDifference"]?.values,[0])
                XCTAssertNotNil(fixture.inputs["parameter.weight"]);XCTAssertNil(fixture.inputs["upstream"])
            }
            XCTAssertTrue(controls.setField("strict",value:false));XCTAssertTrue(controls.apply())
            XCTAssertFalse(lesson.resultIsCurrent);XCTAssertNil(lesson.recipeProjection)
            XCTAssertTrue(controls.setField("strict",value:true));XCTAssertTrue(controls.apply())
            XCTAssertTrue(lesson.resultIsCurrent)
            XCTAssertTrue(lesson.setMode(.assessment));XCTAssertNil(lesson.recipeProjection)
            try await check(lesson,workspace:workspace);XCTAssertNil(lesson.recipeProjection)
            XCTAssertTrue(lesson.save());records.append((task.rawValue,try XCTUnwrap(workspace.activeProjectID),workspace.graph))
        }
        try reopen(records,root:root,storage:storage)
    }
}
