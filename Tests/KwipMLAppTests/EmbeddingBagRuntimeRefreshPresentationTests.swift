import XCTest
import KwipMLCore
@testable import KwipMLApp

final class EmbeddingBagRuntimeRefreshPresentationTests:XCTestCase {
    @MainActor func testCurrentBagResultsOfferRefreshButHistoricalTrainConstructionNeverDoes()throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
        let lesson=TensorLearningStore(workspace:workspace)
        let view=TensorLearningView(lesson:lesson,workspace:workspace)
        XCTAssertFalse(view.showsRuntimeRefresh)
        for operation in EmbeddingBagLearningTask.allCases.map(\.rawValue)+GradientLearningTask.allCases.map(\.rawValue)+[InitializationLearningTask.allCases[0].rawValue] {
            XCTAssertTrue(lesson.start(operation:operation))
            XCTAssertNotNil(lesson.runtimeReadinessError)
            let current = !RetiredTrainingLessonDestination.contains(operation)
            XCTAssertEqual(lesson.canRefreshLessonRuntime,current,operation)
            XCTAssertEqual(view.showsRuntimeRefresh,current,operation)
            XCTAssertEqual(view.showsRuntimeRefresh,lesson.canRefreshLessonRuntime)
        }
        XCTAssertTrue(workspace.beginProject(.init(name:"Ordinary project")))
        XCTAssertFalse(lesson.canRefreshLessonRuntime)
        XCTAssertFalse(view.showsRuntimeRefresh,"A view retained after leaving its lesson must not expose an active refresh action")
    }
}
