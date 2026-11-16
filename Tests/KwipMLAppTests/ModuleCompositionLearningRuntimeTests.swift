import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ModuleCompositionLearningRuntimeTests:XCTestCase {
    private func wait(_ workspace:WorkspaceStore)async throws {
        let deadline=Date().addingTimeInterval(65)
        while workspace.isAssessingLearning && Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
        XCTAssertFalse(workspace.isAssessingLearning,"The real check must finish within its captured allowance")
    }
    func testBothOwnedLessonsAllModesColdReceiptsAndWrongContainerRepair()async throws {
        guard let path=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Enable actual development worker for composition lessons")}
        let repository=URL(fileURLWithPath:path)
        for task in ModuleCompositionLearningTask.allCases {
            let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer {try? FileManager.default.removeItem(at:storage)}
            let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
            let lesson=TensorLearningStore(workspace:workspace)
            XCTAssertTrue(lesson.start(operation:"prepared-"+task.rawValue),lesson.statusMessage)
            await verifyRuntimeForIntegration(workspace)
            let runtime=try XCTUnwrap(workspace.runtime)
            let inventory=try WorkflowRuntimeEvidence.capture(runtime)
            XCTAssertNotNil(workspace.moduleCompositionDiscoveryEvidence)
            XCTAssertNil(lesson.runtimeReadinessError,workspace.capabilityStatus)
            let correct=try XCTUnwrap(lesson.draft?.compositionModel)
            for mode in [LearningMode.guided,.practice,.assessment] {
                XCTAssertTrue(lesson.setMode(mode))
                XCTAssertTrue(lesson.canSubmit,lesson.statusMessage)
                lesson.submit();try await wait(workspace)
                XCTAssertEqual(lesson.grade?.status,.passed,"\(task.rawValue) \(mode): \(lesson.statusMessage)")
                XCTAssertTrue(lesson.resultSaved,lesson.statusMessage)
                XCTAssertTrue(lesson.resultIsCurrent)
                XCTAssertNil(lesson.formativeProjection,"Nested source is not an ordinary forward-graph submission")
                XCTAssertEqual(lesson.compositionResults == nil,mode == .assessment)
                if let measured=lesson.compositionResults {
                    XCTAssertEqual(measured.projection.fixtures.count,4)
                    XCTAssertEqual(measured.metadata.count,4)
                    XCTAssertTrue(measured.metadata.allSatisfy {$0.independentStorage})
                }
                let id=try XCTUnwrap(workspace.activeProjectID)
                let cold=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
                let reopened=TensorLearningStore(workspace:cold)
                XCTAssertTrue(cold.beginProject(workspace.graph,id:id),cold.statusMessage)
                XCTAssertEqual(reopened.grade?.status,.passed,reopened.statusMessage)
                XCTAssertTrue(reopened.resultSaved);XCTAssertTrue(reopened.resultIsCurrent)
                XCTAssertFalse(cold.isAssessingLearning)
                XCTAssertEqual(reopened.compositionResults == nil,mode == .assessment)
                XCTAssertNotNil(reopened.runtimeReadinessError,"Saved results grant no fresh runtime execution")
                XCTAssertEqual(try WorkflowRuntimeEvidence.capture(runtime),inventory)
            }
            if task == .orderName {
                XCTAssertTrue(lesson.setMode(.practice))
                var wrong=correct
                let first=try XCTUnwrap(wrong.source.root.instances.first)
                let index=try XCTUnwrap(wrong.source.definitions.firstIndex {$0.reference == first.definition})
                let children=try XCTUnwrap(wrong.source.definitions[index].composition?.children)
                wrong.source.definitions[index].composition = .init(kind:.graph,children:children)
                XCTAssertTrue(lesson.drafts.update {$0.compositionModel=wrong},lesson.drafts.errorMessage ?? "")
                XCTAssertNil(lesson.compositionResults)
                XCTAssertTrue(lesson.canSubmit,lesson.statusMessage)
                lesson.submit();try await wait(workspace)
                XCTAssertEqual(lesson.grade?.status,.failed,lesson.statusMessage)
                XCTAssertTrue(lesson.grade?.outcomes.contains {$0.criterionID == "structuralIdentity" && !$0.passed} == true)
                XCTAssertTrue(lesson.drafts.update {$0.compositionModel=correct})
                lesson.submit();try await wait(workspace)
                XCTAssertEqual(lesson.grade?.status,.passed,lesson.statusMessage)
            }
        }
    }
}
