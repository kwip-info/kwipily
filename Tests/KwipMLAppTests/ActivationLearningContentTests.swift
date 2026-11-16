import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ActivationLearningContentTests:XCTestCase {
    @MainActor func testOverviewCoversEveryExactFunctionWithoutDropoutAndKeepsLessonRoutes() {
        let entries=ActivationLearningContent.entries
        XCTAssertEqual(Set(entries.map { $0.function.rawValue }),Set(ActivationConfiguration.Function.allCases.map(\.rawValue)))
        XCTAssertEqual(entries.count,10)
        XCTAssertFalse(entries.contains { $0.function.rawValue == "dropout" })
        let routes=["relu":"relu-lesson","sigmoid":"sigmoid-lesson","tanh":"tanh-lesson","gelu":"gelu","silu":"silu","leaky_relu":"leaky_relu","elu":"elu","prelu":"prelu","softmax":"softmax","log_softmax":"log_softmax"]
        for entry in entries {
            XCTAssertEqual(entry.lessonID,routes[entry.function.rawValue])
            XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID:entry.lessonID),entry.lessonID)
            XCTAssertFalse(entry.summary.isEmpty)
        }
        XCTAssertEqual(ActivationLearningContent.defaultFunction,.relu)
    }

    func testPreservesIndependentAssessmentIdentity() {
        XCTAssertEqual(Set(ConfigurableActivationLearningTask.allCases.map(\.rawValue)),["silu-values","leaky-relu-slope","elu-alpha"])
        XCTAssertEqual(Set(RemainingActivationLearningTask.allCases.map(\.rawValue)),["prelu-shared-slope","prelu-channel-slopes","logsoftmax-sample-scores","dropout-training-pattern","dropout-evaluation-pattern"])
    }
}
